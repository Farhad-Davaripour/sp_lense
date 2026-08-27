from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sp_lense.comparison_runtime import next_token_logits, resolve_choice_boundary
from sp_lense.gradient_specificity_trust_region import (
    assess_trial_acceptance,
    linearized_lower_bounds,
    solve_generalized_min_l2_qp,
    terminal_bidirectional_decision_gate,
    trust_step_cap_fraction,
    update_trust_radius,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gradient_specificity_v3_development as base

LOCK_PATH = ROOT / "configs" / "gradient_specificity_trust_region_lock.json"
PROBE_SUMMARY_DEFAULT_PATH = (
    ROOT
    / "results"
    / "gradient_specificity_v3_development"
    / "absolute_dose_probe_v1"
    / "qwen35_08b"
    / "stage_a"
    / "absolute_dose_summary.json"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "gradient_specificity_trust_region_development" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "gradient_specificity_trust_region_development" / "qwen35_08b"
AUDIT_ROWS_PATH = RESULT_ROOT / "score_audit_rows.jsonl"
AUDIT_MANIFEST_PATH = RESULT_ROOT / "score_audit_manifest.json"
SUMMARY_PATH = RESULT_ROOT / "development_summary.json"
REPORT_PATH = RESULT_ROOT / "DEVELOPMENT_REPORT.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_trust_region.py"

LOCK_SCHEMA = "sp_lense.gradient_specificity_trust_region_lock.v1"
PROBE_SUMMARY_SCHEMA = "sp_lense.gradient_specificity_v3_absolute_dose_summary.v1"
CHECKPOINT_SCHEMA = "sp_lense.gradient_specificity_trust_region_checkpoint.v1"
BUDGET_STATE_SCHEMA = "sp_lense.gradient_specificity_trust_region_budget_state.v1"
RESULT_SCHEMA = "sp_lense.gradient_specificity_trust_region_direction_result.v1"
AUDIT_ROWS_SCHEMA = "sp_lense.gradient_specificity_trust_region_audit_rows.v1"
SUMMARY_SCHEMA = "sp_lense.gradient_specificity_trust_region_summary.v1"

EXPECTED_OPTIMIZER_VALUES = {
    "target_margin_logit": 0.01,
    "matched_other_margin_logit": 0.0,
    "maximum_accepted_iterations": 16,
    "maximum_backtracks_per_iteration": 8,
    "acceptance_ratio": 0.1,
    "high_agreement_ratio": 0.75,
    "shrink_factor": 0.5,
    "expansion_factor": 2.0,
    "individual_violation_tolerance": 1e-6,
    "maximum_forward_evaluations_per_direction": 512,
    "maximum_backward_evaluations_per_direction": 128,
}
EXPECTED_PROTECTED_LIMITS = {
    "maximum_exact_greedy_token_changes": 0,
    "maximum_semantic_decision_changes": 0,
    "maximum_new_other_outputs": 0,
    "mean_full_vocabulary_kl_changed_to_baseline": 0.005,
    "empirical_p95_full_vocabulary_kl_changed_to_baseline": 0.02,
    "maximum_full_vocabulary_kl_changed_to_baseline": 0.05,
}
NULL_APPLICATION_TOLERANCE = 2e-5


class ComputeBudgetExhausted(RuntimeError):
    """Raised before a model operation that would exceed a locked direction budget."""


class EvaluationBudget:
    """Count actual model forward/backward calls and fail closed at locked ceilings."""

    def __init__(
        self,
        *,
        maximum_forward_evaluations: int,
        maximum_backward_evaluations: int,
        forward_evaluations: int = 0,
        backward_evaluations: int = 0,
        on_change: Callable[[Mapping[str, int]], None] | None = None,
    ) -> None:
        values = {
            "maximum_forward_evaluations": maximum_forward_evaluations,
            "maximum_backward_evaluations": maximum_backward_evaluations,
            "forward_evaluations": forward_evaluations,
            "backward_evaluations": backward_evaluations,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
            raise TypeError("compute-budget values must be integers")
        if maximum_forward_evaluations < 0 or maximum_backward_evaluations < 0:
            raise ValueError("compute-budget ceilings must be non-negative")
        if not 0 <= forward_evaluations <= maximum_forward_evaluations:
            raise ValueError("forward evaluation count is outside its locked budget")
        if not 0 <= backward_evaluations <= maximum_backward_evaluations:
            raise ValueError("backward evaluation count is outside its locked budget")
        self.maximum_forward_evaluations = maximum_forward_evaluations
        self.maximum_backward_evaluations = maximum_backward_evaluations
        self.forward_evaluations = forward_evaluations
        self.backward_evaluations = backward_evaluations
        self._on_change = on_change

    def require_capacity(self, *, forward: int = 0, backward: int = 0) -> None:
        if (
            isinstance(forward, bool)
            or isinstance(backward, bool)
            or not isinstance(forward, int)
            or not isinstance(backward, int)
            or forward < 0
            or backward < 0
        ):
            raise ValueError("requested evaluation capacity must be non-negative integers")
        if self.forward_evaluations + forward > self.maximum_forward_evaluations:
            raise ComputeBudgetExhausted(
                "next forward evaluation would exceed the locked per-direction ceiling"
            )
        if self.backward_evaluations + backward > self.maximum_backward_evaluations:
            raise ComputeBudgetExhausted(
                "next backward evaluation would exceed the locked per-direction ceiling"
            )

    def record_forward(self) -> None:
        self.require_capacity(forward=1)
        self.forward_evaluations += 1
        self._notify()

    def record_backward(self) -> None:
        self.require_capacity(backward=1)
        self.backward_evaluations += 1
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self.snapshot())

    def set_on_change(
        self,
        callback: Callable[[Mapping[str, int]], None] | None,
    ) -> None:
        self._on_change = callback

    def snapshot(self) -> dict[str, int]:
        return {
            "forward_evaluations": self.forward_evaluations,
            "backward_evaluations": self.backward_evaluations,
            "maximum_forward_evaluations": self.maximum_forward_evaluations,
            "maximum_backward_evaluations": self.maximum_backward_evaluations,
            "remaining_forward_evaluations": (
                self.maximum_forward_evaluations - self.forward_evaluations
            ),
            "remaining_backward_evaluations": (
                self.maximum_backward_evaluations - self.backward_evaluations
            ),
        }


def _budget_from_optimizer(
    optimizer: Mapping[str, Any],
    *,
    stored: Mapping[str, Any] | None = None,
) -> EvaluationBudget:
    stored = {} if stored is None else stored
    return EvaluationBudget(
        maximum_forward_evaluations=int(optimizer["maximum_forward_evaluations_per_direction"]),
        maximum_backward_evaluations=int(optimizer["maximum_backward_evaluations_per_direction"]),
        forward_evaluations=int(stored.get("forward_evaluations", 0)),
        backward_evaluations=int(stored.get("backward_evaluations", 0)),
    )


def canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _float(value: Any, *, field: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _exact_number(observed: Any, wanted: float, *, field: str) -> float:
    value = _float(observed, field=field, nonnegative=wanted == 0.0, positive=wanted > 0.0)
    if not math.isclose(value, wanted, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{field} differs from the locked value {wanted}")
    return value


def _load_lock_and_probe() -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not LOCK_PATH.is_file():
        raise RuntimeError("trust-region execution requires the committed lock file")
    lock = _load_json(LOCK_PATH)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("trust-region lock has the wrong schema")
    if lock.get("status") != "locked_before_trust_region_execution":
        raise ValueError("trust-region protocol was not locked before execution")
    if lock.get("model") != base.EXPECTED_MODEL:
        raise ValueError("trust-region model settings differ from frozen v3")
    if lock.get("development_only") is not True:
        raise ValueError("trust-region lock must be explicitly development-only")

    optimizer = lock.get("optimizer")
    limits = lock.get("protected_limits")
    binding = lock.get("absolute_probe_summary")
    if not isinstance(optimizer, dict) or not isinstance(limits, dict):
        raise TypeError("trust-region lock lacks optimizer or protected_limits")
    if not isinstance(binding, dict):
        raise TypeError("trust-region lock lacks its absolute_probe_summary binding")

    for key, wanted in EXPECTED_OPTIMIZER_VALUES.items():
        observed = optimizer.get(key)
        if isinstance(wanted, int):
            if isinstance(observed, bool) or not isinstance(observed, int) or observed != wanted:
                raise ValueError(f"optimizer.{key} differs from the locked value {wanted}")
        else:
            _exact_number(observed, wanted, field=f"optimizer.{key}")
    for key, wanted in EXPECTED_PROTECTED_LIMITS.items():
        observed = limits.get(key)
        if isinstance(wanted, int):
            if isinstance(observed, bool) or not isinstance(observed, int) or observed != wanted:
                raise ValueError(f"protected_limits.{key} differs from {wanted}")
        else:
            _exact_number(observed, wanted, field=f"protected_limits.{key}")

    cap = _float(
        optimizer.get("absolute_residual_relative_cap"),
        field="optimizer.absolute_residual_relative_cap",
        positive=True,
    )
    radius_rules = {
        "initial_trust_radius": cap / 4.0,
        "maximum_trust_radius": cap / 2.0,
        "minimum_trust_radius": cap / 256.0,
    }
    for key, wanted in radius_rules.items():
        _exact_number(optimizer.get(key), wanted, field=f"optimizer.{key}")

    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("absolute_probe_summary.path must be non-empty")
    probe_path = ROOT / path_value
    if probe_path.resolve() != PROBE_SUMMARY_DEFAULT_PATH.resolve():
        raise ValueError("absolute probe summary path differs from the frozen default")
    if not probe_path.is_file():
        raise RuntimeError("absolute probe summary is not complete")
    expected_file_hash = binding.get("sha256")
    if not isinstance(expected_file_hash, str) or len(expected_file_hash) != 64:
        raise ValueError("absolute_probe_summary.sha256 is malformed")
    if file_sha256(probe_path) != expected_file_hash:
        raise RuntimeError("absolute probe summary differs from the lock binding")
    probe = _load_json(probe_path)
    if (
        probe.get("schema_version") != PROBE_SUMMARY_SCHEMA
        or probe.get("development_only") is not True
        or probe.get("status") != "complete"
        or probe.get("trust_radius_selection_uses_self_outcomes") is not False
    ):
        raise ValueError("absolute probe summary lacks its completed protected-only identity")
    stored_summary_hash = probe.get("summary_sha256")
    recomputed_summary_hash = canonical_sha256(
        {key: value for key, value in probe.items() if key != "summary_sha256"}
    )
    if stored_summary_hash != recomputed_summary_hash:
        raise RuntimeError("absolute probe summary failed its internal hash")
    selected = probe.get("selected_empirical_trust_radius")
    if selected is None or probe.get("no_supported_positive_radius_on_grid") is not False:
        raise RuntimeError("absolute probe found no supported positive optimizer cap")
    selected_value = _float(selected, field="selected_empirical_trust_radius", positive=True)
    bound_selected = _float(
        binding.get("selected_empirical_trust_radius"),
        field="absolute_probe_summary.selected_empirical_trust_radius",
        positive=True,
    )
    if not (
        math.isclose(cap, selected_value, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(cap, bound_selected, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise RuntimeError("optimizer cap differs from the protected-only probe selection")
    return lock, probe, probe_path


def _load_frozen_inputs(torch: Any) -> dict[str, Any]:
    manifest = base.load_development_manifest()
    nuisance_forms = base.render_unrelated_forms("nuisance_fit")
    sp_forms = base.render_sp_forms("A")
    nuisance_identity = base._artifact_identity(
        kind="nuisance_capture", stage=None, forms=nuisance_forms
    )
    nuisance_capture = base._load_complete_capture(
        torch,
        capture_path=base.NUISANCE_CAPTURE_PATH,
        manifest_path=base.NUISANCE_MANIFEST_PATH,
        identity=nuisance_identity,
        forms=nuisance_forms,
    )
    sp_capture_path, sp_manifest_path = base._sp_capture_paths("A")
    sp_identity = base._artifact_identity(kind="sp_capture", stage="A", forms=sp_forms)
    sp_capture = base._load_complete_capture(
        torch,
        capture_path=sp_capture_path,
        manifest_path=sp_manifest_path,
        identity=sp_identity,
        forms=sp_forms,
    )
    bank = base._load_complete_bank("A")
    raw_rows = torch.cat(
        [
            torch.cat(
                (
                    record["semantic_gradient"].reshape(1, -1).double(),
                    record["greedy_competitor_gap_gradients"].double(),
                ),
                dim=0,
            )
            for record in nuisance_capture["records"]
        ],
        dim=0,
    )
    construction = manifest["construction"]
    basis, diagnostics = base.v3.row_normalized_svd_basis(
        torch,
        raw_rows,
        rtol=float(construction["nuisance_svd_relative_tolerance"]),
        atol=float(construction["nuisance_svd_absolute_tolerance"]),
    )
    frozen_global = bank.get("global_nuisance", {})
    if (
        raw_rows.shape != (288, 1024)
        or basis.shape != (255, 1024)
        or diagnostics["basis_sha256"] != frozen_global.get("basis_sha256")
        or int(frozen_global.get("rank", -1)) != 255
    ):
        raise RuntimeError("reconstructed global nuisance basis differs from frozen v3")
    sp_records = {str(record["form_id"]): record for record in sp_capture["records"]}
    nuisance_records = {str(record["form_id"]): record for record in nuisance_capture["records"]}
    if set(sp_records) != {str(form["form_id"]) for form in sp_forms}:
        raise RuntimeError("frozen Stage-A SP record coverage is incomplete")
    if set(nuisance_records) != {str(form["form_id"]) for form in nuisance_forms}:
        raise RuntimeError("frozen nuisance record coverage is incomplete")
    return {
        "manifest": manifest,
        "sp_forms": sp_forms,
        "sp_records": sp_records,
        "nuisance_forms": nuisance_forms,
        "nuisance_records": nuisance_records,
        "global_nuisance_basis": basis.contiguous(),
        "global_nuisance_diagnostics": diagnostics,
        "source_hashes": {
            "nuisance_capture": file_sha256(base.NUISANCE_CAPTURE_PATH),
            "nuisance_capture_manifest": file_sha256(base.NUISANCE_MANIFEST_PATH),
            "sp_capture": file_sha256(sp_capture_path),
            "sp_capture_manifest": file_sha256(sp_manifest_path),
            "direction_bank": file_sha256(base._bank_paths("A")[0]),
            "direction_bank_manifest": file_sha256(base._bank_paths("A")[1]),
        },
    }


def _study_identity(
    *,
    lock: Mapping[str, Any],
    probe_path: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_identity.v1",
        "development_only": True,
        "lock_sha256": file_sha256(LOCK_PATH),
        "absolute_probe_summary_path": _relative(probe_path),
        "absolute_probe_summary_sha256": file_sha256(probe_path),
        "model": lock["model"],
        "optimizer": lock["optimizer"],
        "protected_limits": lock["protected_limits"],
        "source_hashes": dict(frozen["source_hashes"]),
        "global_nuisance_basis_sha256": base.v3.tensor_float64_sha256(
            frozen["global_nuisance_basis"]
        ),
        "global_nuisance_rank": int(frozen["global_nuisance_basis"].shape[0]),
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "math_sha256": file_sha256(MATH_PATH),
        "frozen_v3_runner_sha256": file_sha256(base.SCRIPT_PATH),
        "frozen_v3_math_sha256": file_sha256(base.MATH_MODULE_PATH),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def run_preflight() -> dict[str, Any]:
    lock, _probe, probe_path = _load_lock_and_probe()
    import torch

    frozen = _load_frozen_inputs(torch)
    identity = _study_identity(lock=lock, probe_path=probe_path, frozen=frozen)
    case_assignments = sorted(
        {(str(form["case_id"]), int(form["assignment"])) for form in frozen["sp_forms"]}
    )
    if len(case_assignments) != 8:
        raise RuntimeError("trust-region Stage A must contain eight case assignments")
    output = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_preflight.v1",
        "development_only": True,
        "passes_preflight": True,
        "model_loads": 0,
        "model_forwards": 0,
        "external_model_judges": 0,
        "external_api_calls": 0,
        "case_assignment_attempt_count": len(case_assignments),
        "self_preservation_form_count": len(frozen["sp_forms"]),
        "nuisance_fit_form_count": len(frozen["nuisance_forms"]),
        "constraint_count_per_direction": 8,
        "maximum_forward_evaluations_per_direction": int(
            lock["optimizer"]["maximum_forward_evaluations_per_direction"]
        ),
        "maximum_backward_evaluations_per_direction": int(
            lock["optimizer"]["maximum_backward_evaluations_per_direction"]
        ),
        "hard_equality_source": "frozen_global_unrelated_task_basis_only",
        "matched_other_is_per_iterate_inequality_not_permanent_null": True,
        "identity": identity,
    }
    output["preflight_sha256"] = canonical_sha256(output)
    return output


def load_backend() -> Any:
    return base.load_backend()


def _parameter_grad_count(model: Any) -> int:
    parameters = model.parameters() if hasattr(model, "parameters") else ()
    return sum(parameter.grad is not None for parameter in parameters)


def _semantic_choice(token_id: int, *, positive_id: int, negative_id: int) -> str:
    if token_id == positive_id:
        return "positive"
    if token_id == negative_id:
        return "negative"
    return "OTHER"


def _strongest_competitor(torch: Any, logits: Any, desired_token_id: int) -> tuple[int, Any]:
    if desired_token_id < 0 or desired_token_id >= int(logits.numel()):
        raise ValueError("desired token ID is outside the vocabulary")
    masked = logits.clone()
    masked[desired_token_id] = -torch.inf
    competitor_id = int(torch.argmax(masked).item())
    if competitor_id == desired_token_id or not bool(torch.isfinite(masked[competitor_id]).item()):
        raise RuntimeError("could not resolve a finite strongest vocabulary competitor")
    return competitor_id, logits[desired_token_id] - logits[competitor_id]


def _resolve_ids(backend: Any, form: Mapping[str, Any]) -> tuple[Any, Any, int, int]:
    prompt = str(form["prompt"])
    tokens = backend.encode(prompt)
    boundary = resolve_choice_boundary(backend, prompt)
    if int(tokens.shape[-1]) != int(boundary.prompt_length):
        raise RuntimeError("choice boundary length differs from encoded prompt")
    positive_id = int(boundary.token_id(str(form["positive_label"])))
    negative_id = int(boundary.token_id(str(form["negative_label"])))
    if positive_id == negative_id:
        raise RuntimeError("positive and negative choices resolve to the same token")
    return tokens, boundary, positive_id, negative_id


def _constraint_specifications(
    *,
    case_id: str,
    assignment: int,
    frozen: Mapping[str, Any],
    optimizer: Mapping[str, Any],
) -> list[dict[str, Any]]:
    forms = [
        form
        for form in frozen["sp_forms"]
        if str(form["case_id"]) == case_id and int(form["assignment"]) == assignment
    ]
    forms_by_cell = {(str(form["target"]), bool(form["preserve_first"])): form for form in forms}
    expected = {(target, order) for target in ("self", "other") for order in (True, False)}
    if set(forms_by_cell) != expected:
        raise RuntimeError("case assignment lacks exact self/other and both-order coverage")
    output = []
    for family, target, required_key in (
        ("self", "self", "target_margin_logit"),
        ("matched_other", "other", "matched_other_margin_logit"),
    ):
        for preserve_first in (True, False):
            form = forms_by_cell[(target, preserve_first)]
            record = frozen["sp_records"][str(form["form_id"])]
            if not bool(record["baseline_answer_format_valid"]):
                raise RuntimeError("trust-region fitting requires valid A/B frozen baselines")
            for sign in (1, -1):
                output.append(
                    {
                        "constraint_id": (
                            f"{family}:preserve_{'A' if preserve_first else 'B'}:"
                            f"{'plus' if sign == 1 else 'minus'}"
                        ),
                        "family": family,
                        "target": target,
                        "preserve_first": preserve_first,
                        "sign": sign,
                        "required_margin": float(optimizer[required_key]),
                        "form": form,
                        "frozen_record": record,
                    }
                )
    if len(output) != 8 or [item["family"] for item in output].count("self") != 4:
        raise RuntimeError("trust-region constraint construction did not produce exact 4+4 rows")
    return output


def _baseline_observation(
    backend: Any,
    *,
    form: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    cache: dict[str, dict[str, Any]],
    budget: EvaluationBudget,
) -> dict[str, Any]:
    form_id = str(form["form_id"])
    cached = cache.get(form_id)
    if cached is not None:
        return cached
    tokens, boundary, positive_id, negative_id = _resolve_ids(backend, form)
    budget.require_capacity(forward=1)
    budget.record_forward()
    logits = next_token_logits(backend, tokens).detach().float().cpu().contiguous()
    if logits.ndim != 1 or not bool(backend.torch.isfinite(logits).all().item()):
        raise RuntimeError("baseline next-token logits are not a finite vocabulary vector")
    actual_token_id = int(logits.argmax().item())
    semantic = _semantic_choice(
        actual_token_id,
        positive_id=positive_id,
        negative_id=negative_id,
    )
    expected = {
        "actual_token_id": int(frozen_record["baseline_greedy_token_id"]),
        "semantic": str(frozen_record["baseline_actual_semantic_choice"]),
        "positive_or_a_token_id": (
            int(frozen_record["choice_a_token_id"])
            if str(form["positive_label"]) == "A"
            else int(frozen_record["choice_b_token_id"])
        ),
        "negative_or_b_token_id": (
            int(frozen_record["choice_a_token_id"])
            if str(form["negative_label"]) == "A"
            else int(frozen_record["choice_b_token_id"])
        ),
    }
    if (
        actual_token_id != expected["actual_token_id"]
        or semantic != expected["semantic"]
        or positive_id != expected["positive_or_a_token_id"]
        or negative_id != expected["negative_or_b_token_id"]
        or str(boundary.evidence_sha256) != str(frozen_record["choice_boundary_evidence_sha256"])
    ):
        raise RuntimeError("fresh baseline differs from the frozen v3 capture")
    if semantic == "OTHER" or not bool(frozen_record["baseline_answer_format_valid"]):
        raise RuntimeError("protected baseline must be a valid authored A/B token")
    output = {
        "form_id": form_id,
        "logits": logits,
        "actual_token_id": actual_token_id,
        "semantic": semantic,
        "positive_id": positive_id,
        "negative_id": negative_id,
        "prompt_length": int(tokens.shape[-1]),
        "choice_boundary_evidence_sha256": str(boundary.evidence_sha256),
    }
    cache[form_id] = output
    return output


def _run_logits_with_delta(
    backend: Any,
    *,
    form: Mapping[str, Any],
    delta: Any,
    sign: int,
    layer: int,
    budget: EvaluationBudget,
) -> tuple[Any, dict[str, Any]]:
    if sign not in {-1, 1} or isinstance(sign, bool):
        raise ValueError("intervention sign must be +1 or -1")
    torch = backend.torch
    tokens, _boundary, _positive_id, _negative_id = _resolve_ids(backend, form)
    vector = delta.detach().to(device=tokens.device, dtype=torch.float32).contiguous()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all().item()):
        raise ValueError("delta must be a finite one-dimensional tensor")
    captured: dict[str, Any] = {"hook_calls": 0}

    def intervention_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("trust-region intervention hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise RuntimeError("trust-region residual activation must be [1, sequence, width]")
        if int(activation.shape[1]) != int(tokens.shape[-1]):
            raise RuntimeError("trust-region activation does not end at the prompt boundary")
        if int(activation.shape[-1]) != int(vector.numel()):
            raise RuntimeError("delta width differs from the intervention residual width")
        prompt_index = int(tokens.shape[-1]) - 1
        original = activation.detach().float()
        residual_norm = original[0, prompt_index].norm().detach()
        if not bool(torch.isfinite(residual_norm).item()) or float(residual_norm.item()) <= 0.0:
            raise RuntimeError("prompt-final residual norm is not finite and positive")
        mask = torch.zeros_like(original)
        mask[:, prompt_index, :] = 1.0
        changed = original + sign * residual_norm * mask * vector.view(1, 1, -1)
        returned = changed.to(dtype=activation.dtype)
        returned_float = returned.detach().float()
        if not torch.equal(returned_float[:, :prompt_index], original[:, :prompt_index]):
            raise RuntimeError("trust-region hook changed a non-final prompt position")
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
            raise RuntimeError("trust-region application diagnostics are non-finite")
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
        logits = backend.model(tokens)[0, -1].detach().float().cpu().contiguous()
    if captured["hook_calls"] != 1 or captured.get("selected_position_count") != 1:
        raise RuntimeError("trust-region intervention did not select exactly one position")
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("trust-region changed logits are not finite")
    diagnostics = {
        **captured,
        "sign": sign,
        "requested_delta_norm": float(vector.norm().item()),
        "delta_float32_sha256": base.v3.tensor_float32_sha256(vector.cpu()),
    }
    return logits, diagnostics


def _capture_constraint_observation(
    backend: Any,
    *,
    specification: Mapping[str, Any],
    delta: Any,
    layer: int,
    budget: EvaluationBudget,
) -> dict[str, Any]:
    """Measure one finite constraint and its gradient with respect to delta."""

    torch = backend.torch
    sign = int(specification["sign"])
    if sign not in {-1, 1}:
        raise ValueError("constraint sign must be +1 or -1")
    form = specification["form"]
    tokens, boundary, positive_id, negative_id = _resolve_ids(backend, form)
    if _parameter_grad_count(backend.model) != 0:
        raise RuntimeError("model parameter gradients were populated before capture")
    delta_leaf = (
        delta.detach()
        .to(device=tokens.device, dtype=torch.float32)
        .contiguous()
        .requires_grad_(True)
    )
    if delta_leaf.ndim != 1 or not bool(torch.isfinite(delta_leaf).all().item()):
        raise ValueError("delta must be a finite one-dimensional tensor")
    captured: dict[str, Any] = {"hook_calls": 0}

    def intervention_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        if captured["hook_calls"] != 1:
            raise RuntimeError("trust-region gradient hook fired more than once")
        if activation.ndim != 3 or int(activation.shape[0]) != 1:
            raise RuntimeError("gradient residual activation must be [1, sequence, width]")
        if int(activation.shape[1]) != int(tokens.shape[-1]):
            raise RuntimeError("gradient activation does not end at the prompt boundary")
        if int(activation.shape[-1]) != int(delta_leaf.numel()):
            raise RuntimeError("delta width differs from the gradient residual width")
        prompt_index = int(tokens.shape[-1]) - 1
        original = activation.detach().float()
        residual_norm = original[0, prompt_index].norm().detach()
        if not bool(torch.isfinite(residual_norm).item()) or float(residual_norm.item()) <= 0.0:
            raise RuntimeError("gradient residual norm is not finite and positive")
        mask = torch.zeros_like(original)
        mask[:, prompt_index, :] = 1.0
        changed = original + sign * residual_norm * mask * delta_leaf.view(1, 1, -1)
        if not torch.equal(changed[:, :prompt_index], original[:, :prompt_index]):
            raise RuntimeError("gradient hook changed a non-final prompt position")
        captured.update(
            {
                "selected_position_count": 1,
                "prompt_final_index": prompt_index,
                "residual_norm": float(residual_norm.item()),
            }
        )
        return changed.to(dtype=activation.dtype)

    budget.require_capacity(forward=1, backward=1)
    budget.record_forward()
    with (
        torch.enable_grad(),
        backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", intervention_hook)]),
    ):
        logits = backend.model(tokens)[0, -1].float()
        if specification["family"] == "self":
            desired_id = positive_id if sign == 1 else negative_id
        elif specification["family"] == "matched_other":
            desired_id = int(specification["frozen_record"]["baseline_greedy_token_id"])
        else:
            raise ValueError("unknown trust-region constraint family")
        competitor_id, objective = _strongest_competitor(torch, logits, desired_id)
        budget.record_backward()
        gradient = torch.autograd.grad(
            objective,
            delta_leaf,
            retain_graph=False,
            create_graph=False,
        )[0]
    if captured["hook_calls"] != 1 or captured.get("selected_position_count") != 1:
        raise RuntimeError("trust-region gradient did not select exactly one position")
    if _parameter_grad_count(backend.model) != 0:
        raise RuntimeError("trust-region autograd populated model parameter gradients")
    gradient = gradient.detach().float().cpu().contiguous()
    logits_cpu = logits.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(gradient).all().item()) or not bool(
        torch.isfinite(logits_cpu).all().item()
    ):
        raise RuntimeError("trust-region constraint capture contains a non-finite value")
    actual_token_id = int(logits_cpu.argmax().item())
    semantic_positive_gap = float((logits_cpu[positive_id] - logits_cpu[negative_id]).item())
    desired_semantic_gap = (
        sign * semantic_positive_gap if specification["family"] == "self" else semantic_positive_gap
    )
    return {
        "constraint_id": str(specification["constraint_id"]),
        "family": str(specification["family"]),
        "form_id": str(form["form_id"]),
        "preserve_first": bool(specification["preserve_first"]),
        "sign": sign,
        "required_margin": float(specification["required_margin"]),
        "constraint_value": float(objective.detach().item()),
        "constraint_gradient": gradient,
        "constraint_gradient_sha256": base.v3.tensor_float32_sha256(gradient),
        "desired_token_id": desired_id,
        "strongest_competitor_token_id": competitor_id,
        "actual_token_id": actual_token_id,
        "actual_semantic_choice": _semantic_choice(
            actual_token_id,
            positive_id=positive_id,
            negative_id=negative_id,
        ),
        "semantic_desired_gap": desired_semantic_gap,
        "positive_id": positive_id,
        "negative_id": negative_id,
        "choice_boundary_evidence_sha256": str(boundary.evidence_sha256),
        "selected_position_count": int(captured["selected_position_count"]),
        "residual_norm": float(captured["residual_norm"]),
        "delta_float32_sha256": base.v3.tensor_float32_sha256(delta_leaf.detach().cpu()),
    }


def _changed_observation(
    backend: Any,
    *,
    specification: Mapping[str, Any],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    budget: EvaluationBudget,
) -> dict[str, Any]:
    form = specification["form"]
    frozen_record = specification["frozen_record"]
    baseline = _baseline_observation(
        backend,
        form=form,
        frozen_record=frozen_record,
        cache=baseline_cache,
        budget=budget,
    )
    logits, intervention = _run_logits_with_delta(
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
    if specification["family"] == "self":
        desired_id = positive_id if sign == 1 else negative_id
    else:
        desired_id = int(baseline["actual_token_id"])
    competitor_id, constraint_value = _strongest_competitor(
        backend.torch,
        logits,
        desired_id,
    )
    actual_token_id = int(logits.argmax().item())
    actual_semantic = _semantic_choice(
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
        raise RuntimeError("changed-to-baseline KL is invalid")
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


def _constraint_system(
    backend: Any,
    *,
    specifications: Sequence[Mapping[str, Any]],
    delta: Any,
    layer: int,
    budget: EvaluationBudget,
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    observations = [
        _capture_constraint_observation(
            backend,
            specification=specification,
            delta=delta,
            layer=layer,
            budget=budget,
        )
        for specification in specifications
    ]
    torch = backend.torch
    gradients = torch.stack(
        [observation["constraint_gradient"] for observation in observations]
    ).double()
    values = torch.tensor(
        [observation["constraint_value"] for observation in observations],
        dtype=torch.float64,
    )
    required = torch.tensor(
        [observation["required_margin"] for observation in observations],
        dtype=torch.float64,
    )
    if gradients.shape[0] != 8 or values.shape != (8,) or required.shape != (8,):
        raise RuntimeError("trust-region finite constraint system must contain eight rows")
    return gradients, values, required, observations


def _nearest_rank_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _application_report(
    observations: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("application certification requires at least one observation")
    rows = []
    for row in observations:
        intervention = row.get("intervention")
        if not isinstance(intervention, Mapping):
            raise TypeError("application certification lacks intervention diagnostics")
        requested = float(intervention["requested_relative_perturbation_norm"])
        realized = float(intervention["realized_relative_perturbation_norm"])
        norm_error = float(intervention["absolute_relative_perturbation_error"])
        coordinate_error = float(intervention["maximum_abs_relative_application_coordinate_error"])
        tolerance = NULL_APPLICATION_TOLERANCE * (1.0 + requested)
        if not all(
            math.isfinite(value)
            for value in (requested, realized, norm_error, coordinate_error, tolerance)
        ):
            raise RuntimeError("application certification contains a non-finite value")
        gates = {
            "one_hook_call": intervention.get("hook_calls") == 1,
            "one_selected_position": intervention.get("selected_position_count") == 1,
            "relative_norm_matches_request": norm_error <= tolerance,
            "relative_vector_matches_request": coordinate_error <= tolerance,
        }
        rows.append(
            {
                "constraint_id": str(row["constraint_id"]),
                "requested_relative_norm": requested,
                "realized_relative_norm": realized,
                "absolute_relative_norm_error": norm_error,
                "maximum_abs_relative_coordinate_error": coordinate_error,
                "tolerance": tolerance,
                "gates": gates,
                "passes": all(gates.values()),
            }
        )
    report = {
        "group": group,
        "signed_row_count": len(rows),
        "maximum_absolute_relative_norm_error": max(
            float(row["absolute_relative_norm_error"]) for row in rows
        ),
        "maximum_abs_relative_coordinate_error": max(
            float(row["maximum_abs_relative_coordinate_error"]) for row in rows
        ),
        "rows": rows,
        "passes": all(bool(row["passes"]) for row in rows),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _protection_report(
    observations: Sequence[Mapping[str, Any]],
    *,
    limits: Mapping[str, Any],
    group: str,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("finite protection requires at least one observation")
    exact_changes = sum(bool(row["exact_token_changed"]) for row in observations)
    semantic_changes = sum(bool(row["semantic_decision_changed"]) for row in observations)
    new_other = sum(bool(row["new_other_output"]) for row in observations)
    kl_values = [float(row["full_vocabulary_kl_changed_to_baseline"]) for row in observations]
    if any(not math.isfinite(value) or value < 0.0 for value in kl_values):
        raise RuntimeError("finite protection contains an invalid KL value")
    kl = {
        "row_count": len(kl_values),
        "mean": statistics.fmean(kl_values),
        "empirical_p95": _nearest_rank_quantile(kl_values, 0.95),
        "maximum": max(kl_values),
    }
    application = _application_report(observations, group=group)
    gates = {
        "exact_greedy_token_changes": exact_changes
        <= int(limits["maximum_exact_greedy_token_changes"]),
        "semantic_decision_changes": semantic_changes
        <= int(limits["maximum_semantic_decision_changes"]),
        "new_other_outputs": new_other <= int(limits["maximum_new_other_outputs"]),
        "mean_kl": kl["mean"] <= float(limits["mean_full_vocabulary_kl_changed_to_baseline"]),
        "empirical_p95_kl": kl["empirical_p95"]
        <= float(limits["empirical_p95_full_vocabulary_kl_changed_to_baseline"]),
        "maximum_kl": kl["maximum"]
        <= float(limits["maximum_full_vocabulary_kl_changed_to_baseline"]),
        "exact_single_position_application": bool(application["passes"]),
    }
    report = {
        "group": group,
        "signed_row_count": len(observations),
        "exact_greedy_token_changes": exact_changes,
        "semantic_decision_changes": semantic_changes,
        "new_other_outputs": new_other,
        "full_vocabulary_kl_changed_to_baseline": kl,
        "application": application,
        "gates": gates,
        "passes": all(gates.values()),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _null_certificate(
    torch: Any, *, delta: Any, global_basis: Any, absolute_cap: float
) -> dict[str, Any]:
    candidate = delta.detach().cpu().double().contiguous()
    if candidate.ndim != 1 or int(candidate.numel()) != int(global_basis.shape[1]):
        raise ValueError("delta width differs from the global nuisance basis")
    if not bool(torch.isfinite(candidate).all().item()):
        raise ValueError("delta contains a non-finite value")
    norm = float(torch.linalg.vector_norm(candidate).item())
    applied = candidate.float().double().contiguous()
    applied_norm = float(torch.linalg.vector_norm(applied).item())
    construction_projection = (
        float(torch.max(torch.abs(global_basis @ candidate)).item())
        if global_basis.numel()
        else 0.0
    )
    applied_projection = (
        float(torch.max(torch.abs(global_basis @ applied)).item()) if global_basis.numel() else 0.0
    )
    base_tolerance = max(
        float(base.v3.DEFAULT_RESIDUAL_TOLERANCE),
        8.0 * float(torch.finfo(torch.float32).eps),
    )
    construction_allowed = float(base.v3.DEFAULT_RESIDUAL_TOLERANCE) * (1.0 + norm)
    applied_allowed = base_tolerance * (1.0 + applied_norm)
    cap_allowed = base_tolerance * (1.0 + float(absolute_cap))
    values = (
        norm,
        applied_norm,
        construction_projection,
        applied_projection,
        construction_allowed,
        applied_allowed,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("global nuisance null certificate is non-finite")
    gates = {
        "float64_global_null": construction_projection <= construction_allowed,
        "float32_application_global_null": applied_projection <= applied_allowed,
        "absolute_residual_relative_cap": applied_norm <= float(absolute_cap) + cap_allowed,
    }
    report = {
        "delta_float64_sha256": base.v3.tensor_float64_sha256(candidate),
        "delta_float32_sha256": base.v3.tensor_float32_sha256(candidate.float()),
        "float64_norm": norm,
        "float32_application_norm": applied_norm,
        "absolute_cap": float(absolute_cap),
        "maximum_abs_float64_global_basis_projection": construction_projection,
        "maximum_abs_float32_global_basis_projection": applied_projection,
        "float64_projection_tolerance": construction_allowed,
        "float32_projection_tolerance": applied_allowed,
        "gates": gates,
        "passes": all(gates.values()),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _terminal_from_self_observations(
    backend: Any,
    *,
    observations: Sequence[Mapping[str, Any]],
    target_margin: float,
) -> dict[str, Any]:
    self_rows = [row for row in observations if row["family"] == "self"]
    if len(self_rows) != 4:
        raise RuntimeError("terminal gate requires exactly four signed self rows")
    by_cell = {(bool(row["preserve_first"]), int(row["sign"])): row for row in self_rows}
    expected = {(order, sign) for order in (True, False) for sign in (1, -1)}
    if set(by_cell) != expected:
        raise RuntimeError("terminal self rows lack both signs and both orders")
    semantic = []
    vocabulary = []
    actual = []
    baseline = []
    preserve_ids = []
    comply_ids = []
    for order in (True, False):
        plus = by_cell[(order, 1)]
        minus = by_cell[(order, -1)]
        semantic.append([float(plus["semantic_desired_gap"]), float(minus["semantic_desired_gap"])])
        vocabulary.append([float(plus["constraint_value"]), float(minus["constraint_value"])])
        actual.append([int(plus["actual_token_id"]), int(minus["actual_token_id"])])
        if int(plus["baseline_actual_token_id"]) != int(minus["baseline_actual_token_id"]):
            raise RuntimeError("self baseline token differs across signed evaluations")
        baseline.append(int(plus["baseline_actual_token_id"]))
        preserve_ids.append(int(plus["positive_id"]))
        comply_ids.append(int(plus["negative_id"]))
    torch = backend.torch
    return terminal_bidirectional_decision_gate(
        torch,
        semantic_desired_gaps=torch.tensor(semantic, dtype=torch.float64),
        full_vocabulary_desired_gaps=torch.tensor(vocabulary, dtype=torch.float64),
        actual_token_ids=torch.tensor(actual, dtype=torch.int64),
        baseline_actual_token_ids=torch.tensor(baseline, dtype=torch.int64),
        preserve_token_ids=torch.tensor(preserve_ids, dtype=torch.int64),
        comply_token_ids=torch.tensor(comply_ids, dtype=torch.int64),
        decision_margin=float(target_margin),
    )


def _primary_trial_evaluation(
    backend: Any,
    *,
    specifications: Sequence[Mapping[str, Any]],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    limits: Mapping[str, Any],
    target_margin: float,
    budget: EvaluationBudget,
) -> dict[str, Any]:
    observations = [
        _changed_observation(
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
        raise RuntimeError("primary trial evaluation did not produce eight signed rows")
    self_rows = [row for row in observations if row["family"] == "self"]
    other_rows = [row for row in observations if row["family"] == "matched_other"]
    self_application = _application_report(self_rows, group="self")
    other_report = _protection_report(other_rows, limits=limits, group="matched_other")
    terminal = _terminal_from_self_observations(
        backend,
        observations=observations,
        target_margin=target_margin,
    )
    values = backend.torch.tensor(
        [float(row["constraint_value"]) for row in observations],
        dtype=backend.torch.float64,
    )
    return {
        "values": values,
        "observations": observations,
        "self_application": self_application,
        "matched_other": other_report,
        "terminal_gate": terminal,
    }


def _nuisance_specifications(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for form in frozen["nuisance_forms"]:
        record = frozen["nuisance_records"][str(form["form_id"])]
        if not bool(record["baseline_answer_format_valid"]):
            raise RuntimeError("nuisance finite protection requires valid frozen A/B baselines")
        for sign in (1, -1):
            output.append(
                {
                    "constraint_id": f"nuisance:{form['form_id']}:{'plus' if sign == 1 else 'minus'}",
                    "family": "matched_other",
                    "form": form,
                    "frozen_record": record,
                    "preserve_first": bool(form.get("preferred_first", True)),
                    "sign": sign,
                    "required_margin": 0.0,
                }
            )
    if len(output) != 64:
        raise RuntimeError("nuisance finite protection must contain exactly 64 signed rows")
    return output


def _nuisance_trial_evaluation(
    backend: Any,
    *,
    frozen: Mapping[str, Any],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    limits: Mapping[str, Any],
    budget: EvaluationBudget,
) -> dict[str, Any]:
    observations = [
        _changed_observation(
            backend,
            specification=specification,
            delta=delta,
            layer=layer,
            baseline_cache=baseline_cache,
            budget=budget,
        )
        for specification in _nuisance_specifications(frozen)
    ]
    return {
        "report": _protection_report(observations, limits=limits, group="nuisance_fit"),
        "observations": observations,
    }


def _observation_log(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key not in {"logits", "constraint_gradient"}
    }


def _evaluate_trial(
    backend: Any,
    *,
    specifications: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    delta: Any,
    layer: int,
    baseline_cache: dict[str, dict[str, Any]],
    limits: Mapping[str, Any],
    target_margin: float,
    global_basis: Any,
    absolute_cap: float,
    current_target_values: Any,
    predicted_target_values: Any,
    required_target_values: Any,
    acceptance_ratio: float,
    individual_violation_tolerance: float,
    budget: EvaluationBudget,
) -> dict[str, Any]:
    """Evaluate primary rows first and skip nuisance work after an early failure."""

    primary = _primary_trial_evaluation(
        backend,
        specifications=specifications,
        delta=delta,
        layer=layer,
        baseline_cache=baseline_cache,
        limits=limits,
        target_margin=target_margin,
        budget=budget,
    )
    null_report = _null_certificate(
        backend.torch,
        delta=delta,
        global_basis=global_basis,
        absolute_cap=absolute_cap,
    )
    primary_protection_passed = (
        bool(primary["self_application"]["passes"])
        and bool(primary["matched_other"]["passes"])
        and bool(null_report["passes"])
    )
    primary_acceptance = assess_trial_acceptance(
        backend.torch,
        current_values=current_target_values,
        predicted_trial_values=predicted_target_values,
        measured_trial_values=primary["values"][:4],
        required_lower_bounds=required_target_values,
        finite_protection_passed=primary_protection_passed,
        minimum_acceptance_ratio=acceptance_ratio,
        individual_violation_tolerance=individual_violation_tolerance,
    )
    if not bool(primary_acceptance["accepted"]):
        return {
            "primary": primary,
            "null_certificate": null_report,
            "acceptance": primary_acceptance,
            "nuisance_evaluated": False,
            "nuisance": None,
            "finite_protection_passed": False,
        }
    nuisance = _nuisance_trial_evaluation(
        backend,
        frozen=frozen,
        delta=delta,
        layer=layer,
        baseline_cache=baseline_cache,
        limits=limits,
        budget=budget,
    )
    return {
        "primary": primary,
        "null_certificate": null_report,
        "nuisance_evaluated": True,
        "acceptance": assess_trial_acceptance(
            backend.torch,
            current_values=current_target_values,
            predicted_trial_values=predicted_target_values,
            measured_trial_values=primary["values"][:4],
            required_lower_bounds=required_target_values,
            finite_protection_passed=(
                primary_protection_passed and bool(nuisance["report"]["passes"])
            ),
            minimum_acceptance_ratio=acceptance_ratio,
            individual_violation_tolerance=individual_violation_tolerance,
        ),
        "nuisance": nuisance,
        "finite_protection_passed": (
            primary_protection_passed and bool(nuisance["report"]["passes"])
        ),
    }


def _direction_key(case_id: str, assignment: int) -> str:
    return base._direction_key(case_id, assignment)


def _direction_root(direction_key: str) -> Path:
    return ARTIFACT_ROOT / "directions" / canonical_sha256(direction_key)[:24]


def _direction_identity(
    *,
    study_identity: Mapping[str, Any],
    case_id: str,
    assignment: int,
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = [
        {
            "constraint_id": str(item["constraint_id"]),
            "family": str(item["family"]),
            "form_id": str(item["form"]["form_id"]),
            "prompt_sha256": str(item["form"]["prompt_sha256"]),
            "preserve_first": bool(item["preserve_first"]),
            "sign": int(item["sign"]),
            "required_margin": float(item["required_margin"]),
            "frozen_semantic_gradient_sha256": str(
                item["frozen_record"]["semantic_gradient_sha256"]
            ),
        }
        for item in specifications
    ]
    identity = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_direction_identity.v1",
        "development_only": True,
        "study_identity_sha256": str(study_identity["identity_sha256"]),
        "direction_key": _direction_key(case_id, assignment),
        "case_id": case_id,
        "assignment": assignment,
        "constraint_manifest": manifest,
        "constraint_manifest_sha256": canonical_sha256(manifest),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _checkpoint_paths(root: Path, accepted_iteration: int) -> tuple[Path, Path]:
    stem = f"accepted_{accepted_iteration:03d}"
    return root / "checkpoints" / f"{stem}.pt", root / "checkpoints" / f"{stem}.json"


def _budget_state_path(root: Path) -> Path:
    return root / "compute_budget_state.json"


def _load_budget_state(
    *,
    root: Path,
    identity: Mapping[str, Any],
    optimizer: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _budget_state_path(root)
    if not path.exists():
        return None
    state = _load_json(path)
    if (
        state.get("schema_version") != BUDGET_STATE_SCHEMA
        or state.get("development_only") is not True
        or state.get("direction_identity_sha256") != identity["identity_sha256"]
        or canonical_sha256({key: value for key, value in state.items() if key != "state_sha256"})
        != state.get("state_sha256")
        or not isinstance(state.get("compute_budget"), dict)
    ):
        raise RuntimeError("compute-budget state failed identity validation")
    budget = _budget_from_optimizer(optimizer, stored=state["compute_budget"])
    if state["compute_budget"] != budget.snapshot():
        raise RuntimeError("compute-budget state differs from the locked ceilings")
    return state


def _write_budget_state(
    *,
    root: Path,
    identity: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    compute_budget: Mapping[str, Any],
) -> None:
    budget = _budget_from_optimizer(optimizer, stored=compute_budget)
    if dict(compute_budget) != budget.snapshot():
        raise RuntimeError("refusing to persist a malformed compute-budget state")
    existing = _load_budget_state(root=root, identity=identity, optimizer=optimizer)
    if existing is not None:
        old = existing["compute_budget"]
        if int(compute_budget["forward_evaluations"]) < int(old["forward_evaluations"]) or int(
            compute_budget["backward_evaluations"]
        ) < int(old["backward_evaluations"]):
            raise RuntimeError("compute-budget counters may not decrease")
    state = {
        "schema_version": BUDGET_STATE_SCHEMA,
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "counting_rule": (
            "a slot is consumed and persisted immediately before each model forward or "
            "autograd backward operation"
        ),
        "compute_budget": dict(compute_budget),
    }
    state["state_sha256"] = canonical_sha256(state)
    base.atomic_json(_budget_state_path(root), state)


def _attach_budget_journal(
    budget: EvaluationBudget,
    *,
    root: Path,
    identity: Mapping[str, Any],
    optimizer: Mapping[str, Any],
) -> None:
    _write_budget_state(
        root=root,
        identity=identity,
        optimizer=optimizer,
        compute_budget=budget.snapshot(),
    )
    budget.set_on_change(
        lambda snapshot: _write_budget_state(
            root=root,
            identity=identity,
            optimizer=optimizer,
            compute_budget=snapshot,
        )
    )


def _save_checkpoint(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
    accepted_iteration: int,
    delta: Any,
    trust_radius: float,
    last_trial: Mapping[str, Any] | None,
    compute_budget: Mapping[str, Any],
) -> dict[str, Any]:
    tensor_path, manifest_path = _checkpoint_paths(root, accepted_iteration)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "accepted_iteration": accepted_iteration,
        "delta": delta.detach().cpu().double().contiguous(),
        "trust_radius": float(trust_radius),
        "last_trial": None if last_trial is None else dict(last_trial),
        "compute_budget": dict(compute_budget),
    }
    payload["delta_sha256"] = base.v3.tensor_float64_sha256(payload["delta"])
    payload["payload_identity_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "delta"}
    )
    if tensor_path.exists() or manifest_path.exists():
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("checkpoint tensor and manifest must exist together")
        existing = torch.load(tensor_path, map_location="cpu", weights_only=False)
        manifest = _load_json(manifest_path)
        if (
            file_sha256(tensor_path) != manifest.get("checkpoint_file_sha256")
            or existing.get("payload_identity_sha256") != payload["payload_identity_sha256"]
            or not torch.equal(existing.get("delta"), payload["delta"])
        ):
            raise RuntimeError("existing accepted checkpoint differs from deterministic state")
        return manifest
    base.atomic_torch_save(torch, tensor_path, payload)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "accepted_iteration": accepted_iteration,
        "checkpoint_path": _relative(tensor_path),
        "checkpoint_file_sha256": file_sha256(tensor_path),
        "delta_sha256": payload["delta_sha256"],
        "trust_radius": float(trust_radius),
        "compute_budget": dict(compute_budget),
        "payload_identity_sha256": payload["payload_identity_sha256"],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    base.atomic_json(manifest_path, manifest)
    return manifest


def _load_latest_checkpoint(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
    dimension: int,
) -> dict[str, Any] | None:
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    manifests = sorted(checkpoint_root.glob("accepted_*.json"))
    tensors = sorted(checkpoint_root.glob("accepted_*.pt"))
    if not manifests:
        if tensors:
            raise RuntimeError("checkpoint directory contains tensors without manifests")
        return None
    if {path.stem for path in manifests} != {path.stem for path in tensors}:
        raise RuntimeError("checkpoint tensor/manifest coverage differs")
    iterations = []
    latest = None
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        iteration = int(manifest.get("accepted_iteration", -1))
        iterations.append(iteration)
        tensor_path, expected_manifest_path = _checkpoint_paths(root, iteration)
        if expected_manifest_path != manifest_path or not tensor_path.is_file():
            raise RuntimeError("checkpoint filename does not match its accepted iteration")
        if (
            manifest.get("schema_version") != CHECKPOINT_SCHEMA
            or manifest.get("development_only") is not True
            or manifest.get("direction_identity_sha256") != identity["identity_sha256"]
            or file_sha256(tensor_path) != manifest.get("checkpoint_file_sha256")
            or canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            != manifest.get("manifest_sha256")
        ):
            raise RuntimeError("checkpoint manifest differs from the direction identity")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        delta = payload.get("delta")
        recomputed_payload_identity = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key not in {"delta", "payload_identity_sha256"}
            }
        )
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA
            or payload.get("direction_identity_sha256") != identity["identity_sha256"]
            or int(payload.get("accepted_iteration", -1)) != iteration
            or not torch.is_tensor(delta)
            or delta.dtype != torch.float64
            or tuple(delta.shape) != (dimension,)
            or base.v3.tensor_float64_sha256(delta) != payload.get("delta_sha256")
            or payload.get("delta_sha256") != manifest.get("delta_sha256")
            or not isinstance(payload.get("compute_budget"), dict)
            or payload.get("compute_budget") != manifest.get("compute_budget")
            or payload.get("payload_identity_sha256") != recomputed_payload_identity
            or payload.get("payload_identity_sha256") != manifest.get("payload_identity_sha256")
        ):
            raise RuntimeError("checkpoint payload failed validation")
        latest = payload
    if iterations != list(range(iterations[-1] + 1)):
        raise RuntimeError("accepted checkpoint sequence is not contiguous from zero")
    return latest


def _uncheckpointed_work_after(
    *,
    root: Path,
    identity: Mapping[str, Any],
    accepted_iteration: int,
    optimizer: Mapping[str, Any],
    checkpoint_budget: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Detect an interrupted partial iteration and refuse unsafe replay."""

    evidence = []
    budget_snapshots = [dict(checkpoint_budget)]
    state = _load_budget_state(root=root, identity=identity, optimizer=optimizer)
    if state is not None:
        state_budget = dict(state["compute_budget"])
        if int(state_budget["forward_evaluations"]) < int(
            checkpoint_budget["forward_evaluations"]
        ) or int(state_budget["backward_evaluations"]) < int(
            checkpoint_budget["backward_evaluations"]
        ):
            raise RuntimeError("compute-budget state predates the latest accepted checkpoint")
        if int(state_budget["forward_evaluations"]) > int(
            checkpoint_budget["forward_evaluations"]
        ) or int(state_budget["backward_evaluations"]) > int(
            checkpoint_budget["backward_evaluations"]
        ):
            evidence.append(_relative(_budget_state_path(root)))
            budget_snapshots.append(state_budget)
    linearization_tensor = root / "linearizations" / f"at_accepted_{accepted_iteration:03d}.pt"
    linearization_manifest = linearization_tensor.with_suffix(".json")
    if linearization_tensor.exists() or linearization_manifest.exists():
        if not linearization_tensor.is_file() or not linearization_manifest.is_file():
            raise RuntimeError("interrupted linearization tensor/manifest coverage differs")
        manifest = _load_json(linearization_manifest)
        if manifest.get("direction_identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("interrupted linearization has the wrong direction identity")
        evidence.append(_relative(linearization_manifest))
        if isinstance(manifest.get("compute_budget"), dict):
            budget_snapshots.append(dict(manifest["compute_budget"]))

    trial_root = root / "trials"
    if trial_root.is_dir():
        for path in sorted(trial_root.glob("iteration_*_backtrack_*.json")):
            trial = _load_json(path)
            if trial.get("direction_identity_sha256") != identity["identity_sha256"]:
                raise RuntimeError("trial log has the wrong direction identity")
            if int(trial.get("accepted_iteration_before_trial", -1)) < accepted_iteration:
                continue
            evidence.append(_relative(path))
            for key in ("compute_budget_before_trial", "compute_budget_after_trial"):
                if isinstance(trial.get(key), dict):
                    budget_snapshots.append(dict(trial[key]))
    if not evidence:
        return None

    validated = [_budget_from_optimizer(optimizer, stored=value) for value in budget_snapshots]
    maximum_forward = max(item.forward_evaluations for item in validated)
    maximum_backward = max(item.backward_evaluations for item in validated)
    budget = _budget_from_optimizer(
        optimizer,
        stored={
            "forward_evaluations": maximum_forward,
            "backward_evaluations": maximum_backward,
        },
    )
    return {
        "reason": "interrupted_after_uncheckpointed_model_evaluations",
        "evidence_paths": evidence,
        "compute_budget": budget.snapshot(),
    }


def _save_linearization(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
    accepted_iteration: int,
    delta: Any,
    gradients: Any,
    values: Any,
    required: Any,
    observations: Sequence[Mapping[str, Any]],
    compute_budget: Mapping[str, Any],
) -> dict[str, Any]:
    directory = root / "linearizations"
    tensor_path = directory / f"at_accepted_{accepted_iteration:03d}.pt"
    manifest_path = directory / f"at_accepted_{accepted_iteration:03d}.json"
    payload = {
        "accepted_iteration": accepted_iteration,
        "delta": delta.detach().cpu().double().contiguous(),
        "gradient_rows": gradients.detach().cpu().double().contiguous(),
        "current_values": values.detach().cpu().double().contiguous(),
        "required_values": required.detach().cpu().double().contiguous(),
    }
    tensor_hashes = {
        "delta": base.v3.tensor_float64_sha256(payload["delta"]),
        "gradient_rows": base.v3.tensor_float64_sha256(payload["gradient_rows"]),
        "current_values": base.v3.tensor_float64_sha256(payload["current_values"]),
        "required_values": base.v3.tensor_float64_sha256(payload["required_values"]),
    }
    manifest = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_linearization.v1",
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "accepted_iteration": accepted_iteration,
        "tensor_hashes": tensor_hashes,
        "observations": [_observation_log(row) for row in observations],
        "compute_budget": dict(compute_budget),
    }
    manifest["linearization_identity_sha256"] = canonical_sha256(manifest)
    if tensor_path.exists() or manifest_path.exists():
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("linearization tensor and manifest must exist together")
        existing = torch.load(tensor_path, map_location="cpu", weights_only=False)
        existing_manifest = _load_json(manifest_path)
        if (
            existing_manifest.get("linearization_identity_sha256")
            != manifest["linearization_identity_sha256"]
            or file_sha256(tensor_path) != existing_manifest.get("tensor_file_sha256")
            or canonical_sha256(
                {key: value for key, value in existing_manifest.items() if key != "manifest_sha256"}
            )
            != existing_manifest.get("manifest_sha256")
            or int(existing.get("accepted_iteration", -1)) != accepted_iteration
            or any(
                not torch.equal(existing[key], payload[key])
                for key in ("delta", "gradient_rows", "current_values", "required_values")
            )
        ):
            raise RuntimeError("existing linearization differs from recomputation")
        return existing_manifest
    base.atomic_torch_save(torch, tensor_path, payload)
    manifest["tensor_path"] = _relative(tensor_path)
    manifest["tensor_file_sha256"] = file_sha256(tensor_path)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    base.atomic_json(manifest_path, manifest)
    return manifest


def _trial_log_value(trial: Mapping[str, Any]) -> dict[str, Any]:
    primary = trial["primary"]
    nuisance = trial.get("nuisance")
    return {
        "primary": {
            "values": [float(value) for value in primary["values"].tolist()],
            "observations": [_observation_log(row) for row in primary["observations"]],
            "self_application": primary["self_application"],
            "matched_other": primary["matched_other"],
            "terminal_gate": primary["terminal_gate"],
        },
        "null_certificate": trial["null_certificate"],
        "acceptance": trial["acceptance"],
        "nuisance_evaluated": bool(trial["nuisance_evaluated"]),
        "nuisance": (
            None
            if nuisance is None
            else {
                "report": nuisance["report"],
                "observations": [_observation_log(row) for row in nuisance["observations"]],
            }
        ),
        "finite_protection_passed": bool(trial["finite_protection_passed"]),
    }


def _save_trial_log(
    *,
    root: Path,
    identity: Mapping[str, Any],
    accepted_iteration_before_trial: int,
    backtrack: int,
    value: Mapping[str, Any],
) -> Path:
    path = (
        root
        / "trials"
        / f"iteration_{accepted_iteration_before_trial + 1:03d}_backtrack_{backtrack:02d}.json"
    )
    payload = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_trial.v1",
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "accepted_iteration_before_trial": accepted_iteration_before_trial,
        "backtrack": backtrack,
        **dict(value),
    }
    payload["trial_sha256"] = canonical_sha256(payload)
    if path.exists():
        if _load_json(path) != payload:
            raise RuntimeError("existing immutable trial log differs from recomputation")
    else:
        base.atomic_json(path, payload)
    return path


def _result_paths(root: Path) -> tuple[Path, Path]:
    return root / "direction_result.pt", root / "direction_result.json"


def _finalize_direction_result(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
    status: str,
    reason: str,
    accepted_iteration: int,
    trust_radius: float,
    delta: Any | None,
    terminal_trial: Mapping[str, Any] | None,
    compute_budget: Mapping[str, Any],
) -> dict[str, Any]:
    if status not in {
        "success",
        "infeasible",
        "failed_numerical",
        "compute_budget_exhausted",
    }:
        raise ValueError("unsupported trust-region terminal status")
    tensor_path, manifest_path = _result_paths(root)
    payload = {
        "schema_version": RESULT_SCHEMA,
        "development_only": True,
        "direction_identity": dict(identity),
        "status": status,
        "reason": reason,
        "accepted_iteration": accepted_iteration,
        "trust_radius": float(trust_radius),
        "delta": None if delta is None else delta.detach().cpu().double().contiguous(),
        "terminal_trial": None if terminal_trial is None else dict(terminal_trial),
        "compute_budget": dict(compute_budget),
    }
    delta_hash = (
        None if payload["delta"] is None else base.v3.tensor_float64_sha256(payload["delta"])
    )
    public = {
        key: value for key, value in payload.items() if key not in {"delta", "terminal_trial"}
    }
    public.update(
        {
            "delta_sha256": delta_hash,
            "has_publishable_direction": status == "success",
            "terminal_trial": payload["terminal_trial"],
        }
    )
    if status == "success" and (payload["delta"] is None or terminal_trial is None):
        raise RuntimeError("successful trust-region result lacks a delta or terminal certificate")
    if status != "success" and public["has_publishable_direction"]:
        raise RuntimeError("failed trust-region result cannot expose a direction")
    public["result_identity_sha256"] = canonical_sha256(public)
    payload["result_identity_sha256"] = public["result_identity_sha256"]
    if tensor_path.exists() or manifest_path.exists():
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("direction result tensor and manifest must exist together")
        existing_public = _load_json(manifest_path)
        if existing_public.get("result_identity_sha256") != public["result_identity_sha256"]:
            raise RuntimeError("existing direction result differs from terminal state")
        return existing_public
    base.atomic_torch_save(torch, tensor_path, payload)
    public["result_tensor_path"] = _relative(tensor_path)
    public["result_tensor_file_sha256"] = file_sha256(tensor_path)
    public["manifest_sha256"] = canonical_sha256(public)
    base.atomic_json(manifest_path, public)
    return public


def _load_completed_result(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    tensor_path, manifest_path = _result_paths(root)
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("completed direction tensor and manifest must exist together")
    manifest = _load_json(manifest_path)
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    delta = payload.get("delta")
    payload_public = {
        key: value
        for key, value in payload.items()
        if key not in {"delta", "terminal_trial", "result_identity_sha256"}
    }
    payload_public.update(
        {
            "delta_sha256": (None if delta is None else base.v3.tensor_float64_sha256(delta)),
            "has_publishable_direction": payload.get("status") == "success",
            "terminal_trial": payload.get("terminal_trial"),
        }
    )
    if (
        manifest.get("schema_version") != RESULT_SCHEMA
        or manifest.get("development_only") is not True
        or payload.get("schema_version") != RESULT_SCHEMA
        or payload.get("direction_identity", {}).get("identity_sha256")
        != identity["identity_sha256"]
        or manifest.get("direction_identity", {}).get("identity_sha256")
        != identity["identity_sha256"]
        or file_sha256(tensor_path) != manifest.get("result_tensor_file_sha256")
        or payload.get("result_identity_sha256") != manifest.get("result_identity_sha256")
        or payload.get("compute_budget") != manifest.get("compute_budget")
        or canonical_sha256(payload_public) != payload.get("result_identity_sha256")
        or canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        != manifest.get("manifest_sha256")
    ):
        raise RuntimeError("completed direction result failed identity validation")
    if manifest.get("status") == "success":
        if (
            not torch.is_tensor(delta)
            or delta.dtype != torch.float64
            or base.v3.tensor_float64_sha256(delta) != manifest.get("delta_sha256")
            or manifest.get("has_publishable_direction") is not True
        ):
            raise RuntimeError("successful direction result lacks its certified delta")
    elif delta is not None or manifest.get("has_publishable_direction") is not False:
        raise RuntimeError("failed direction result exposes an uncertified delta")
    return manifest


def _load_completed_result_payload(
    torch: Any,
    *,
    root: Path,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_completed_result(torch, root=root, identity=identity)
    if manifest is None:
        raise RuntimeError("direction optimization result is incomplete")
    tensor_path, _manifest_path = _result_paths(root)
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    return manifest, payload


def _optimize_direction(
    backend: Any,
    *,
    lock: Mapping[str, Any],
    frozen: Mapping[str, Any],
    study_identity: Mapping[str, Any],
    case_id: str,
    assignment: int,
    baseline_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    torch = backend.torch
    optimizer = lock["optimizer"]
    limits = lock["protected_limits"]
    layer = int(lock["model"]["layer_zero_based"])
    cap = float(optimizer["absolute_residual_relative_cap"])
    specifications = _constraint_specifications(
        case_id=case_id,
        assignment=assignment,
        frozen=frozen,
        optimizer=optimizer,
    )
    identity = _direction_identity(
        study_identity=study_identity,
        case_id=case_id,
        assignment=assignment,
        specifications=specifications,
    )
    root = _direction_root(str(identity["direction_key"]))
    completed = _load_completed_result(torch, root=root, identity=identity)
    if completed is not None:
        return completed

    dimension = int(frozen["global_nuisance_basis"].shape[1])
    checkpoint = _load_latest_checkpoint(
        torch,
        root=root,
        identity=identity,
        dimension=dimension,
    )
    if checkpoint is None:
        budget = _budget_from_optimizer(optimizer)
        delta = torch.zeros(dimension, dtype=torch.float64)
        accepted_iteration = 0
        trust_radius = float(optimizer["initial_trust_radius"])
        _save_checkpoint(
            torch,
            root=root,
            identity=identity,
            accepted_iteration=0,
            delta=delta,
            trust_radius=trust_radius,
            last_trial=None,
            compute_budget=budget.snapshot(),
        )
        _attach_budget_journal(
            budget,
            root=root,
            identity=identity,
            optimizer=optimizer,
        )
    else:
        stored_budget = checkpoint.get("compute_budget")
        if not isinstance(stored_budget, dict):
            raise RuntimeError("accepted checkpoint lacks its compute-budget counters")
        budget = _budget_from_optimizer(optimizer, stored=stored_budget)
        if stored_budget != budget.snapshot():
            raise RuntimeError("accepted checkpoint compute budget differs from the lock")
        delta = checkpoint["delta"].detach().cpu().double().contiguous()
        accepted_iteration = int(checkpoint["accepted_iteration"])
        trust_radius = float(checkpoint["trust_radius"])
        interrupted = _uncheckpointed_work_after(
            root=root,
            identity=identity,
            accepted_iteration=accepted_iteration,
            optimizer=optimizer,
            checkpoint_budget=stored_budget,
        )
        if interrupted is not None:
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="failed_numerical",
                reason=str(interrupted["reason"]),
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=None,
                terminal_trial={"interruption_evidence_paths": list(interrupted["evidence_paths"])},
                compute_budget=interrupted["compute_budget"],
            )
        _attach_budget_journal(
            budget,
            root=root,
            identity=identity,
            optimizer=optimizer,
        )
        last_trial = checkpoint.get("last_trial")
        if (
            isinstance(last_trial, dict)
            and last_trial.get("primary", {}).get("terminal_gate", {}).get("passes_terminal_gate")
            is True
            and last_trial.get("finite_protection_passed") is True
        ):
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="success",
                reason="resumed_terminal_checkpoint",
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=delta,
                terminal_trial=last_trial,
                compute_budget=budget.snapshot(),
            )

    maximum_iterations = int(optimizer["maximum_accepted_iterations"])
    maximum_backtracks = int(optimizer["maximum_backtracks_per_iteration"])
    while accepted_iteration < maximum_iterations:
        try:
            gradients, values, required, observations = _constraint_system(
                backend,
                specifications=specifications,
                delta=delta,
                layer=layer,
                budget=budget,
            )
        except ComputeBudgetExhausted as error:
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="compute_budget_exhausted",
                reason=str(error),
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=None,
                terminal_trial=None,
                compute_budget=budget.snapshot(),
            )
        except (RuntimeError, ValueError) as error:
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="failed_numerical",
                reason=f"constraint_capture_failed:{error}",
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=None,
                terminal_trial=None,
                compute_budget=budget.snapshot(),
            )
        _save_linearization(
            torch,
            root=root,
            identity=identity,
            accepted_iteration=accepted_iteration,
            delta=delta,
            gradients=gradients,
            values=values,
            required=required,
            observations=observations,
            compute_budget=budget.snapshot(),
        )
        try:
            absolute_bounds, bound_diagnostics = linearized_lower_bounds(
                torch,
                current_point=delta,
                current_values=values,
                gradient_rows=gradients,
                required_lower_bounds=required,
            )
            proposed, qp_diagnostics = solve_generalized_min_l2_qp(
                torch,
                inequality_rows=gradients,
                lower_bounds=absolute_bounds,
                nuisance_rows=frozen["global_nuisance_basis"],
                svd_rtol=float(
                    frozen["manifest"]["construction"]["nuisance_svd_relative_tolerance"]
                ),
                svd_atol=float(
                    frozen["manifest"]["construction"]["nuisance_svd_absolute_tolerance"]
                ),
            )
        except RuntimeError as error:
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="infeasible",
                reason=f"linearized_subproblem_failed:{error}",
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=None,
                terminal_trial=None,
                compute_budget=budget.snapshot(),
            )

        accepted_this_iteration = False
        for backtrack in range(maximum_backtracks):
            fraction, fraction_diagnostics = trust_step_cap_fraction(
                torch,
                current_point=delta,
                proposed_point=proposed,
                trust_radius=trust_radius,
                absolute_cap=cap,
            )
            trial_delta = (delta + fraction * (proposed - delta)).contiguous()
            predicted_values = values + gradients @ (trial_delta - delta)
            common_trial = {
                "current_delta_sha256": base.v3.tensor_float64_sha256(delta),
                "proposed_delta_sha256": base.v3.tensor_float64_sha256(proposed),
                "trial_delta_sha256": base.v3.tensor_float64_sha256(trial_delta),
                "trust_radius_before_trial": trust_radius,
                "linearized_bound_diagnostics": bound_diagnostics,
                "qp_diagnostics": qp_diagnostics,
                "trust_fraction_diagnostics": fraction_diagnostics,
                "compute_budget_before_trial": budget.snapshot(),
                "predicted_constraint_values": [
                    float(value) for value in predicted_values.tolist()
                ],
            }
            if bool(fraction_diagnostics["zero_usable_fraction"]):
                trial_value = {
                    **common_trial,
                    "accepted": False,
                    "reason": "zero_usable_trust_or_cap_fraction",
                    "finite_evaluation_performed": False,
                }
                _save_trial_log(
                    root=root,
                    identity=identity,
                    accepted_iteration_before_trial=accepted_iteration,
                    backtrack=backtrack,
                    value=trial_value,
                )
                break

            try:
                trial = _evaluate_trial(
                    backend,
                    specifications=specifications,
                    frozen=frozen,
                    delta=trial_delta,
                    layer=layer,
                    baseline_cache=baseline_cache,
                    limits=limits,
                    target_margin=float(optimizer["target_margin_logit"]),
                    global_basis=frozen["global_nuisance_basis"],
                    absolute_cap=cap,
                    current_target_values=values[:4],
                    predicted_target_values=predicted_values[:4],
                    required_target_values=required[:4],
                    acceptance_ratio=float(optimizer["acceptance_ratio"]),
                    individual_violation_tolerance=float(
                        optimizer["individual_violation_tolerance"]
                    ),
                    budget=budget,
                )
            except ComputeBudgetExhausted as error:
                _save_trial_log(
                    root=root,
                    identity=identity,
                    accepted_iteration_before_trial=accepted_iteration,
                    backtrack=backtrack,
                    value={
                        **common_trial,
                        "accepted": False,
                        "reason": "compute_budget_exhausted",
                        "error": str(error),
                        "finite_evaluation_performed": True,
                        "compute_budget_after_trial": budget.snapshot(),
                    },
                )
                return _finalize_direction_result(
                    torch,
                    root=root,
                    identity=identity,
                    status="compute_budget_exhausted",
                    reason=str(error),
                    accepted_iteration=accepted_iteration,
                    trust_radius=trust_radius,
                    delta=None,
                    terminal_trial=None,
                    compute_budget=budget.snapshot(),
                )
            except (RuntimeError, ValueError) as error:
                _save_trial_log(
                    root=root,
                    identity=identity,
                    accepted_iteration_before_trial=accepted_iteration,
                    backtrack=backtrack,
                    value={
                        **common_trial,
                        "accepted": False,
                        "reason": "finite_evaluation_failed",
                        "error": str(error),
                        "finite_evaluation_performed": True,
                        "compute_budget_after_trial": budget.snapshot(),
                    },
                )
                return _finalize_direction_result(
                    torch,
                    root=root,
                    identity=identity,
                    status="failed_numerical",
                    reason=f"finite_evaluation_failed:{error}",
                    accepted_iteration=accepted_iteration,
                    trust_radius=trust_radius,
                    delta=None,
                    terminal_trial=None,
                    compute_budget=budget.snapshot(),
                )
            accepted = bool(trial["acceptance"]["accepted"])
            trial_value = {
                **common_trial,
                "accepted": accepted,
                "reason": str(trial["acceptance"]["reason"]),
                "finite_evaluation_performed": True,
                "compute_budget_after_trial": budget.snapshot(),
                "finite_trial": _trial_log_value(trial),
            }
            _save_trial_log(
                root=root,
                identity=identity,
                accepted_iteration_before_trial=accepted_iteration,
                backtrack=backtrack,
                value=trial_value,
            )
            ratio = trial["acceptance"]["actual_to_predicted_reduction_ratio"]
            next_radius = update_trust_radius(
                current_radius=trust_radius,
                minimum_radius=float(optimizer["minimum_trust_radius"]),
                maximum_radius=float(optimizer["maximum_trust_radius"]),
                accepted=accepted,
                actual_to_predicted_ratio=None if ratio is None else float(ratio),
                step_was_trust_limited=bool(fraction_diagnostics["trust_limited"]),
                expansion_threshold=float(optimizer["high_agreement_ratio"]),
                shrink_factor=float(optimizer["shrink_factor"]),
                expansion_factor=float(optimizer["expansion_factor"]),
            )
            if not accepted:
                if math.isclose(next_radius, trust_radius, rel_tol=0.0, abs_tol=1e-15):
                    break
                trust_radius = next_radius
                continue

            delta = trial_delta
            trust_radius = next_radius
            accepted_iteration += 1
            terminal_trial = _trial_log_value(trial)
            _save_checkpoint(
                torch,
                root=root,
                identity=identity,
                accepted_iteration=accepted_iteration,
                delta=delta,
                trust_radius=trust_radius,
                last_trial=terminal_trial,
                compute_budget=budget.snapshot(),
            )
            accepted_this_iteration = True
            if bool(trial["primary"]["terminal_gate"]["passes_terminal_gate"]) and bool(
                trial["finite_protection_passed"]
            ):
                return _finalize_direction_result(
                    torch,
                    root=root,
                    identity=identity,
                    status="success",
                    reason="terminal_gate_and_finite_protections_passed",
                    accepted_iteration=accepted_iteration,
                    trust_radius=trust_radius,
                    delta=delta,
                    terminal_trial=terminal_trial,
                    compute_budget=budget.snapshot(),
                )
            break

        if not accepted_this_iteration:
            return _finalize_direction_result(
                torch,
                root=root,
                identity=identity,
                status="infeasible",
                reason="no_acceptable_trial_within_locked_backtracking_budget",
                accepted_iteration=accepted_iteration,
                trust_radius=trust_radius,
                delta=None,
                terminal_trial=None,
                compute_budget=budget.snapshot(),
            )

    return _finalize_direction_result(
        torch,
        root=root,
        identity=identity,
        status="infeasible",
        reason="maximum_accepted_iterations_exhausted_without_terminal_success",
        accepted_iteration=accepted_iteration,
        trust_radius=trust_radius,
        delta=None,
        terminal_trial=None,
        compute_budget=budget.snapshot(),
    )


def run_optimize(backend: Any | None = None) -> list[dict[str, Any]]:
    lock, _probe, probe_path = _load_lock_and_probe()
    import torch

    frozen = _load_frozen_inputs(torch)
    identity = _study_identity(lock=lock, probe_path=probe_path, frozen=frozen)
    resident = load_backend() if backend is None else backend
    baseline_cache: dict[str, dict[str, Any]] = {}
    case_assignments = sorted(
        {(str(form["case_id"]), int(form["assignment"])) for form in frozen["sp_forms"]}
    )
    results = []
    for index, (case_id, assignment) in enumerate(case_assignments, start=1):
        print(
            f"TR-CNOG optimize {index}/{len(case_assignments)}: "
            f"{_direction_key(case_id, assignment)}",
            flush=True,
        )
        results.append(
            _optimize_direction(
                resident,
                lock=lock,
                frozen=frozen,
                study_identity=identity,
                case_id=case_id,
                assignment=assignment,
                baseline_cache=baseline_cache,
            )
        )
    return results


def _audit_row_for_success(
    backend: Any,
    *,
    lock: Mapping[str, Any],
    frozen: Mapping[str, Any],
    identity: Mapping[str, Any],
    result_manifest: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    baseline_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    torch = backend.torch
    optimizer = lock["optimizer"]
    limits = lock["protected_limits"]
    budget = _budget_from_optimizer(optimizer)
    delta = result_payload.get("delta")
    if not torch.is_tensor(delta) or delta.dtype != torch.float64:
        raise RuntimeError("score audit received no certified float64 direction")
    try:
        primary = _primary_trial_evaluation(
            backend,
            specifications=specifications,
            delta=delta,
            layer=int(lock["model"]["layer_zero_based"]),
            baseline_cache=baseline_cache,
            limits=limits,
            target_margin=float(optimizer["target_margin_logit"]),
            budget=budget,
        )
        nuisance = _nuisance_trial_evaluation(
            backend,
            frozen=frozen,
            delta=delta,
            layer=int(lock["model"]["layer_zero_based"]),
            baseline_cache=baseline_cache,
            limits=limits,
            budget=budget,
        )
        null_report = _null_certificate(
            torch,
            delta=delta,
            global_basis=frozen["global_nuisance_basis"],
            absolute_cap=float(optimizer["absolute_residual_relative_cap"]),
        )
    except ComputeBudgetExhausted as error:
        return {
            "schema_version": AUDIT_ROWS_SCHEMA,
            "development_only": True,
            "direction_identity_sha256": str(identity["identity_sha256"]),
            "direction_key": str(identity["direction_key"]),
            "optimization_result_identity_sha256": str(result_manifest["result_identity_sha256"]),
            "optimization_status": "success",
            "audit_status": "compute_budget_exhausted",
            "passes_independent_score_audit": False,
            "reason": str(error),
            "compute_budget": budget.snapshot(),
        }
    except (RuntimeError, ValueError) as error:
        return {
            "schema_version": AUDIT_ROWS_SCHEMA,
            "development_only": True,
            "direction_identity_sha256": str(identity["identity_sha256"]),
            "direction_key": str(identity["direction_key"]),
            "optimization_result_identity_sha256": str(result_manifest["result_identity_sha256"]),
            "optimization_status": "success",
            "audit_status": "failed_numerical",
            "passes_independent_score_audit": False,
            "reason": str(error),
            "compute_budget": budget.snapshot(),
        }
    passes = all(
        (
            bool(primary["terminal_gate"]["passes_terminal_gate"]),
            bool(primary["self_application"]["passes"]),
            bool(primary["matched_other"]["passes"]),
            bool(nuisance["report"]["passes"]),
            bool(null_report["passes"]),
        )
    )
    return {
        "schema_version": AUDIT_ROWS_SCHEMA,
        "development_only": True,
        "direction_identity_sha256": str(identity["identity_sha256"]),
        "direction_key": str(identity["direction_key"]),
        "optimization_result_identity_sha256": str(result_manifest["result_identity_sha256"]),
        "optimization_status": "success",
        "audit_status": "passed" if passes else "failed_finite_recheck",
        "passes_independent_score_audit": passes,
        "compute_budget": budget.snapshot(),
        "delta_sha256": str(result_manifest["delta_sha256"]),
        "terminal_gate": primary["terminal_gate"],
        "self_application": primary["self_application"],
        "matched_other": primary["matched_other"],
        "nuisance_fit": nuisance["report"],
        "null_certificate": null_report,
        "primary_observations": [
            _observation_log(observation) for observation in primary["observations"]
        ],
        "nuisance_observations": [
            _observation_log(observation) for observation in nuisance["observations"]
        ],
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    if path.exists():
        if _load_json(path) != payload:
            raise RuntimeError(f"existing immutable artifact differs: {_relative(path)}")
        return
    base.atomic_json(path, payload)


def run_score_audit(backend: Any | None = None) -> list[dict[str, Any]]:
    """Independently rerun every successful direction's complete finite gate."""

    lock, _probe, probe_path = _load_lock_and_probe()
    import torch

    frozen = _load_frozen_inputs(torch)
    study_identity = _study_identity(lock=lock, probe_path=probe_path, frozen=frozen)
    baseline_cache: dict[str, dict[str, Any]] = {}
    resident = backend
    rows = []
    case_assignments = sorted(
        {(str(form["case_id"]), int(form["assignment"])) for form in frozen["sp_forms"]}
    )
    for case_id, assignment in case_assignments:
        specifications = _constraint_specifications(
            case_id=case_id,
            assignment=assignment,
            frozen=frozen,
            optimizer=lock["optimizer"],
        )
        identity = _direction_identity(
            study_identity=study_identity,
            case_id=case_id,
            assignment=assignment,
            specifications=specifications,
        )
        root = _direction_root(str(identity["direction_key"]))
        result_manifest, result_payload = _load_completed_result_payload(
            torch,
            root=root,
            identity=identity,
        )
        if result_manifest["status"] != "success":
            row = {
                "schema_version": AUDIT_ROWS_SCHEMA,
                "development_only": True,
                "direction_identity_sha256": str(identity["identity_sha256"]),
                "direction_key": str(identity["direction_key"]),
                "optimization_result_identity_sha256": str(
                    result_manifest["result_identity_sha256"]
                ),
                "optimization_status": str(result_manifest["status"]),
                "audit_status": "not_applicable_no_successful_direction",
                "passes_independent_score_audit": False,
                "compute_budget": {
                    **_budget_from_optimizer(lock["optimizer"]).snapshot(),
                },
            }
        else:
            if resident is None:
                resident = load_backend()
            row = _audit_row_for_success(
                resident,
                lock=lock,
                frozen=frozen,
                identity=identity,
                result_manifest=result_manifest,
                result_payload=result_payload,
                specifications=specifications,
                baseline_cache=baseline_cache,
            )
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)

    serialized = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    if AUDIT_ROWS_PATH.exists():
        if AUDIT_ROWS_PATH.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("existing immutable score-audit rows differ from recheck")
    else:
        _atomic_text(AUDIT_ROWS_PATH, serialized)
    manifest = {
        "schema_version": "sp_lense.gradient_specificity_trust_region_audit_manifest.v1",
        "development_only": True,
        "study_identity_sha256": str(study_identity["identity_sha256"]),
        "row_count": len(rows),
        "rows_path": _relative(AUDIT_ROWS_PATH),
        "rows_file_sha256": file_sha256(AUDIT_ROWS_PATH),
        "passed_audit_count": sum(bool(row["passes_independent_score_audit"]) for row in rows),
        "model_judges": 0,
        "external_api_calls": 0,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_immutable_json(AUDIT_MANIFEST_PATH, manifest)
    return rows


def _build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gradient-specificity trust-region development result",
        "",
        "Status: development-only; this is not a confirmatory or publication result.",
        "",
        (
            f"The optimizer attempted {summary['direction_attempt_count']} frozen Stage-A "
            f"case/assignment directions. {summary['optimization_success_count']} passed "
            "inside optimization and "
            f"{summary['independently_audited_success_count']} passed an independent finite "
            "recheck."
        ),
        "",
        "| Direction | Optimizer status | Independent audit | Forwards | Backwards |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["directions"]:
        budget = row["optimization_compute_budget"]
        lines.append(
            f"| {row['direction_key']} | {row['optimization_status']} | "
            f"{row['audit_status']} | {budget['forward_evaluations']} | "
            f"{budget['backward_evaluations']} |"
        )
    lines.extend(
        [
            "",
            "A successful row means one bounded residual edit produced the required opposite",
            "self decisions under both answer orders while leaving the matched-other and frozen",
            "nuisance decisions unchanged within the locked KL limits. It does not establish a",
            "natural self-preservation mechanism, general capability preservation, or novelty.",
            "",
        ]
    )
    return "\n".join(lines)


def run_report() -> dict[str, Any]:
    lock, _probe, probe_path = _load_lock_and_probe()
    import torch

    frozen = _load_frozen_inputs(torch)
    study_identity = _study_identity(lock=lock, probe_path=probe_path, frozen=frozen)
    if not AUDIT_ROWS_PATH.is_file() or not AUDIT_MANIFEST_PATH.is_file():
        raise RuntimeError("report requires a completed independent score audit")
    audit_manifest = _load_json(AUDIT_MANIFEST_PATH)
    if (
        audit_manifest.get("study_identity_sha256") != study_identity["identity_sha256"]
        or audit_manifest.get("rows_file_sha256") != file_sha256(AUDIT_ROWS_PATH)
        or canonical_sha256(
            {key: value for key, value in audit_manifest.items() if key != "manifest_sha256"}
        )
        != audit_manifest.get("manifest_sha256")
    ):
        raise RuntimeError("score-audit manifest failed its identity checks")
    audit_rows = _read_jsonl(AUDIT_ROWS_PATH)
    if len(audit_rows) != 8 or int(audit_manifest.get("row_count", -1)) != len(audit_rows):
        raise RuntimeError("score audit lacks exact eight-direction coverage")
    audit_by_key = {str(row["direction_key"]): row for row in audit_rows}
    if len(audit_by_key) != len(audit_rows):
        raise RuntimeError("score audit contains duplicate direction keys")

    directions = []
    for case_id, assignment in sorted(
        {(str(form["case_id"]), int(form["assignment"])) for form in frozen["sp_forms"]}
    ):
        specifications = _constraint_specifications(
            case_id=case_id,
            assignment=assignment,
            frozen=frozen,
            optimizer=lock["optimizer"],
        )
        identity = _direction_identity(
            study_identity=study_identity,
            case_id=case_id,
            assignment=assignment,
            specifications=specifications,
        )
        result, _payload = _load_completed_result_payload(
            torch,
            root=_direction_root(str(identity["direction_key"])),
            identity=identity,
        )
        audit = audit_by_key.get(str(identity["direction_key"]))
        if (
            audit is None
            or audit.get("direction_identity_sha256") != identity["identity_sha256"]
            or audit.get("optimization_result_identity_sha256") != result["result_identity_sha256"]
            or canonical_sha256({key: value for key, value in audit.items() if key != "row_sha256"})
            != audit.get("row_sha256")
        ):
            raise RuntimeError("score-audit row differs from its direction result")
        directions.append(
            {
                "direction_key": str(identity["direction_key"]),
                "case_id": case_id,
                "assignment": assignment,
                "optimization_status": str(result["status"]),
                "optimization_reason": str(result["reason"]),
                "optimization_compute_budget": dict(result["compute_budget"]),
                "audit_status": str(audit["audit_status"]),
                "passes_independent_score_audit": bool(audit["passes_independent_score_audit"]),
                "delta_sha256": result.get("delta_sha256"),
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "development_only": True,
        "status": "complete",
        "study_identity_sha256": str(study_identity["identity_sha256"]),
        "direction_attempt_count": len(directions),
        "optimization_success_count": sum(
            row["optimization_status"] == "success" for row in directions
        ),
        "independently_audited_success_count": sum(
            row["passes_independent_score_audit"] for row in directions
        ),
        "compute_budget_exhausted_count": sum(
            row["optimization_status"] == "compute_budget_exhausted" for row in directions
        ),
        "all_successes_pass_independent_audit": all(
            row["optimization_status"] != "success" or row["passes_independent_score_audit"]
            for row in directions
        ),
        "directions": directions,
        "claim_boundaries": {
            "natural_self_preservation_mechanism_established": False,
            "general_capability_preservation_established": False,
            "publication_ready_novelty_established": False,
        },
        "external_model_judges": 0,
        "external_api_calls": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if SUMMARY_PATH.exists() and _load_json(SUMMARY_PATH) != summary:
        raise RuntimeError("existing immutable summary differs from certified inputs")
    if not SUMMARY_PATH.exists():
        base.atomic_json(SUMMARY_PATH, summary)
    markdown = _build_report(summary)
    if REPORT_PATH.exists():
        if REPORT_PATH.read_text(encoding="utf-8") != markdown:
            raise RuntimeError("existing immutable report differs from certified inputs")
    else:
        _atomic_text(REPORT_PATH, markdown)
    return summary


def _print_command_result(command: str, result: Any) -> None:
    if command == "optimize":
        output = {
            "development_only": True,
            "status": "complete",
            "direction_count": len(result),
            "status_counts": {
                status: sum(row["status"] == status for row in result)
                for status in (
                    "success",
                    "infeasible",
                    "failed_numerical",
                    "compute_budget_exhausted",
                )
            },
        }
    elif command == "score-audit":
        output = {
            "development_only": True,
            "status": "complete",
            "row_count": len(result),
            "passed_audit_count": sum(
                bool(row["passes_independent_score_audit"]) for row in result
            ),
        }
    else:
        output = result
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Development-only trust-region gradient-specificity runner"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "optimize", "score-audit", "report", "all"),
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = run_preflight()
    elif arguments.command == "optimize":
        result = run_optimize()
    elif arguments.command == "score-audit":
        result = run_score_audit()
    elif arguments.command == "report":
        result = run_report()
    else:
        preflight = run_preflight()
        if preflight["passes_preflight"] is not True:
            raise RuntimeError("trust-region preflight did not pass")
        backend = load_backend()
        run_optimize(backend)
        run_score_audit(backend)
        result = run_report()
    _print_command_result(arguments.command, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
