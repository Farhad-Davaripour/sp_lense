"""Finite, opened-development calibration helpers for Decision-Margin Shielding.

This module deliberately contains no model loader and no pilot-set entry point.  It
turns the frozen layer-screen construction into three scenario-local directions,
builds the exact 1,800-forward A/B calibration plan, and scores finite rows under a
strict selectivity gate.  Logit movement is kept separate from an unrestricted
next-token decision change throughout.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .decision_margin_shield import DEFAULT_MARGIN, decision_margin_bounds
from .decision_margin_shield_rowspace import (
    solve_certified_rowspace_minimum_l2_direction,
)
from .factorial_causal_anchor import (
    canonical_sha256,
    render_choice_form,
    render_unrelated_ab_form,
    render_unrelated_construction_form,
    text_sha256,
    validate_pilot_dataset,
)

SCHEMA_VERSION = "sp_lense.decision_margin_shield_finite.v1"
SELECTED_LAYER = 0
SCREEN_RESULT_SHA256 = (
    "4f54383b1d690e7745a9299906385c69d8990f5cd30c973bb322789a5a92b0be"
)
METHODS = ("target_only", "unrelated_null", "decision_margin_shield")
SCREEN_METHODS = {
    "target_only": "unshielded",
    "unrelated_null": "unrelated_only",
    "decision_margin_shield": "decision_margin_shield",
}
STRENGTHS = (0.5, 0.75, 1.0)
SIGNS = (1, -1)
KL_LIMITS = {"mean": 0.005, "p95": 0.02, "max": 0.05}
BASELINE_LOG_ODDS_TOLERANCE = 5e-5
COMPARISON_TOLERANCE = 1e-8
FLOAT32_RAW_CONSTRAINT_TOLERANCE = 2e-5
HOOK_REALIZATION_RELATIVE_L2_TOLERANCE = 1e-4
LEGACY_BAD_CONTROL_ID = "fcag_control_08_instruction"
KL_DOUBLE_ROUNDOFF_FLOOR = -1e-12


@dataclass(frozen=True)
class FiniteDirection:
    scenario_id: str
    method: str
    layer: int
    residual_scale: float
    standardized_direction: np.ndarray
    physical_direction: np.ndarray
    standardized_l2: float
    screen_method: str
    screen_record_sha256: str
    direction_sha256: str
    physical_float32_sha256: str
    solver_diagnostics: Mapping[str, Any]


def _finite_matrix(value: Any, *, field: str, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] < 1:
        raise ValueError(f"{field} must have exactly {rows} non-empty rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite")
    return result


def _finite_vector(value: Any, *, field: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must have exactly {length} finite values")
    return result


def array_float64_sha256(value: Any) -> str:
    """Hash an array using the exact row-space/screen canonical-list convention."""

    canonical = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    return canonical_sha256(canonical.tolist())


def array_float32_sha256(value: Any) -> str:
    canonical = np.asarray(value, dtype=np.float32, order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _screen_record(
    screen_result: Mapping[str, Any], *, scenario_id: str, method: str
) -> Mapping[str, Any]:
    if screen_result.get("result_sha256") != SCREEN_RESULT_SHA256:
        raise RuntimeError("finite construction is bound to a different screen result")
    selection = screen_result.get("selection")
    if not isinstance(selection, Mapping) or selection.get("selected_layer") != SELECTED_LAYER:
        raise RuntimeError("finite construction requires the frozen selected layer 0")
    matched = [
        row
        for row in screen_result.get("geometry_records", [])
        if int(row.get("layer", -1)) == SELECTED_LAYER
        and row.get("scenario_id") == scenario_id
        and row.get("method") == SCREEN_METHODS[method]
    ]
    if len(matched) != 1:
        raise RuntimeError("screen result lacks one exact selected-layer geometry record")
    record = matched[0]
    if record.get("status") != "eligible":
        raise RuntimeError("finite construction cannot use an ineligible screen direction")
    unhashed = dict(record)
    observed = unhashed.pop("screen_record_sha256", None)
    if canonical_sha256(unhashed) != observed:
        raise RuntimeError("screen geometry record self-hash differs")
    return record


def reconstruct_scenario_directions(
    *,
    scenario_id: str,
    residual_scale: float,
    target_rows: Any,
    target_offsets: Any,
    protected_rows: Any,
    protected_offsets: Any,
    unrelated_rows: Any,
    screen_result: Mapping[str, Any],
) -> dict[str, FiniteDirection]:
    """Re-solve and hash-check all three selected-layer scenario directions.

    Inputs are already in residual-relative coordinates.  The returned physical
    vector is ``residual_scale * standardized_direction`` and is the exact absolute
    edit used at the causal anchor before multiplying by sign and strength.
    """

    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if not math.isfinite(float(residual_scale)) or float(residual_scale) <= 0.0:
        raise ValueError("residual_scale must be finite and positive")
    target = _finite_matrix(target_rows, field="target_rows", rows=4)
    target_b = _finite_vector(target_offsets, field="target_offsets", length=4)
    protected = _finite_matrix(protected_rows, field="protected_rows", rows=12)
    protected_b = _finite_vector(
        protected_offsets, field="protected_offsets", length=12
    )
    unrelated = _finite_matrix(unrelated_rows, field="unrelated_rows", rows=8)
    if target.shape[1] != protected.shape[1] or target.shape[1] != unrelated.shape[1]:
        raise ValueError("all finite-construction rows must have equal width")
    protected_bounds = decision_margin_bounds(protected_b, margin=DEFAULT_MARGIN)
    definitions = {
        "target_only": (None, np.zeros(0, dtype=np.float64)),
        "unrelated_null": (unrelated, np.zeros(8, dtype=np.float64)),
        "decision_margin_shield": (
            np.vstack((unrelated, protected)),
            np.concatenate((np.zeros(8, dtype=np.float64), protected_bounds)),
        ),
    }
    result: dict[str, FiniteDirection] = {}
    for method in METHODS:
        nuisance_rows, nuisance_bounds = definitions[method]
        solution = solve_certified_rowspace_minimum_l2_direction(
            target,
            target_b,
            margin=DEFAULT_MARGIN,
            nuisance_rows=nuisance_rows,
            nuisance_bound=nuisance_bounds,
        )
        standardized = np.asarray(
            solution.direction, dtype=np.float64, order="C"
        ).copy(order="C")
        standardized_l2 = float(np.linalg.norm(standardized))
        screen = _screen_record(screen_result, scenario_id=scenario_id, method=method)
        direction_sha = array_float64_sha256(standardized)
        if direction_sha != screen.get("direction_sha256"):
            raise RuntimeError("reconstructed direction hash differs from the screen")
        if not math.isclose(
            standardized_l2,
            float(screen["minimum_standardized_l2"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise RuntimeError("reconstructed direction norm differs from the screen")
        if not math.isclose(
            float(residual_scale),
            float(screen["residual_scale"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("reconstructed residual scale differs from the screen")
        row_hashes = {
            "target_rows_sha256": canonical_sha256(target.tolist()),
            "protected_rows_sha256": canonical_sha256(protected.tolist()),
            "unrelated_rows_sha256": canonical_sha256(unrelated.tolist()),
            "target_offsets_sha256": canonical_sha256(target_b.tolist()),
            "protected_offsets_sha256": canonical_sha256(protected_b.tolist()),
        }
        if any(screen.get(key) != value for key, value in row_hashes.items()):
            raise RuntimeError("reconstructed constraint rows differ from the screen")
        physical = (float(residual_scale) * standardized).astype(np.float32)
        result[method] = FiniteDirection(
            scenario_id=scenario_id,
            method=method,
            layer=SELECTED_LAYER,
            residual_scale=float(residual_scale),
            standardized_direction=standardized,
            physical_direction=physical,
            standardized_l2=standardized_l2,
            screen_method=SCREEN_METHODS[method],
            screen_record_sha256=str(screen["screen_record_sha256"]),
            direction_sha256=direction_sha,
            physical_float32_sha256=array_float32_sha256(physical),
            solver_diagnostics=dict(solution.diagnostics),
        )
    return result


def _deployment_constraint_report(
    standardized: np.ndarray,
    *,
    method: str,
    target_rows: np.ndarray,
    target_offsets: np.ndarray,
    protected_rows: np.ndarray,
    protected_offsets: np.ndarray,
    unrelated_rows: np.ndarray,
    require_target_lower_bound: bool,
) -> dict[str, Any]:
    target_lower = np.abs(target_offsets) + DEFAULT_MARGIN
    target_values = target_rows @ standardized
    maximum_target_violation = (
        float(np.maximum(target_lower - target_values, 0.0).max())
        if require_target_lower_bound
        else None
    )
    maximum_exact_unrelated_residual = (
        0.0
        if method == "target_only"
        else float(np.abs(unrelated_rows @ standardized).max())
    )
    maximum_protected_slab_violation = 0.0
    if method == "decision_margin_shield":
        bounds = decision_margin_bounds(protected_offsets, margin=DEFAULT_MARGIN)
        maximum_protected_slab_violation = float(
            np.maximum(np.abs(protected_rows @ standardized) - bounds, 0.0).max()
        )
    tolerance = FLOAT32_RAW_CONSTRAINT_TOLERANCE
    report = {
        "maximum_target_lower_bound_violation": maximum_target_violation,
        "target_lower_bound_required": require_target_lower_bound,
        "maximum_abs_exact_unrelated_projection": maximum_exact_unrelated_residual,
        "maximum_protected_slab_violation": maximum_protected_slab_violation,
        "raw_log_odds_tolerance": tolerance,
        "exact_cancellation_claim": (
            "not_applicable"
            if method == "target_only"
            else "within_locked_float32_numerical_tolerance"
        ),
    }
    report["passes"] = bool(
        (maximum_target_violation is None or maximum_target_violation <= tolerance)
        and maximum_exact_unrelated_residual <= tolerance
        and maximum_protected_slab_violation <= tolerance
    )
    report["report_sha256"] = canonical_sha256(report)
    return report


def deployment_recertificate(
    direction: FiniteDirection,
    *,
    target_rows: Any,
    target_offsets: Any,
    protected_rows: Any,
    protected_offsets: Any,
    unrelated_rows: Any,
    captured_anchor_residuals: Any,
) -> dict[str, Any]:
    """Certify the stored float32 edit and simulated float32 residual additions.

    The tolerance is inherited from the locked FCAGS deployment precedent; it was not
    selected from these directions.  This certificate does not change or replace the
    original float64 minimum/certificate and does not inspect finite model behavior.
    """

    target = _finite_matrix(target_rows, field="target_rows", rows=4)
    target_b = _finite_vector(target_offsets, field="target_offsets", length=4)
    protected = _finite_matrix(protected_rows, field="protected_rows", rows=12)
    protected_b = _finite_vector(
        protected_offsets, field="protected_offsets", length=12
    )
    unrelated = _finite_matrix(unrelated_rows, field="unrelated_rows", rows=8)
    residuals = np.asarray(
        captured_anchor_residuals, dtype=np.float32, order="C"
    ).copy(order="C")
    if residuals.ndim != 2 or residuals.shape[1] != target.shape[1]:
        raise ValueError("captured_anchor_residuals must have shape [rows, d_model]")
    if residuals.shape[0] < 1 or not np.isfinite(residuals).all():
        raise ValueError("captured anchor residuals must be non-empty and finite")
    physical = np.asarray(
        direction.physical_direction, dtype=np.float32, order="C"
    ).copy(order="C")
    if physical.shape != (target.shape[1],):
        raise ValueError("physical direction width differs from constraints")
    cast_standardized = physical.astype(np.float64) / float(direction.residual_scale)
    stored_report = _deployment_constraint_report(
        cast_standardized,
        method=direction.method,
        target_rows=target,
        target_offsets=target_b,
        protected_rows=protected,
        protected_offsets=protected_b,
        unrelated_rows=unrelated,
        require_target_lower_bound=True,
    )
    simulations = []
    for residual_index, residual in enumerate(residuals):
        for strength in STRENGTHS:
            for sign in SIGNS:
                requested = np.asarray(sign * float(strength) * physical, dtype=np.float32)
                realized = np.asarray((residual + requested) - residual, dtype=np.float32)
                oriented = (
                    np.asarray(sign * realized, dtype=np.float64)
                    / float(direction.residual_scale)
                )
                requested_l2 = float(np.linalg.norm(requested.astype(np.float64)))
                realization_error_l2 = float(
                    np.linalg.norm((realized - requested).astype(np.float64))
                )
                report = _deployment_constraint_report(
                    oriented,
                    method=direction.method,
                    target_rows=target,
                    target_offsets=target_b,
                    protected_rows=protected,
                    protected_offsets=protected_b,
                    unrelated_rows=unrelated,
                    require_target_lower_bound=math.isclose(float(strength), 1.0),
                )
                simulations.append(
                    {
                        "residual_index": residual_index,
                        "strength": float(strength),
                        "sign": sign,
                        "requested_minus_realized_relative_l2": float(
                            realization_error_l2 / max(requested_l2, 1e-12)
                        ),
                        "realized_physical_float32_sha256": array_float32_sha256(realized),
                        "constraints": report,
                    }
                )
    result = {
        "schema_version": f"{SCHEMA_VERSION}.float32_deployment_certificate",
        "scenario_id": direction.scenario_id,
        "method": direction.method,
        "layer": direction.layer,
        "source_float64_direction_sha256": direction.direction_sha256,
        "physical_float32_sha256": direction.physical_float32_sha256,
        "input_hashes": {
            "target_rows_sha256": canonical_sha256(target.tolist()),
            "target_offsets_sha256": canonical_sha256(target_b.tolist()),
            "protected_rows_sha256": canonical_sha256(protected.tolist()),
            "protected_offsets_sha256": canonical_sha256(protected_b.tolist()),
            "unrelated_rows_sha256": canonical_sha256(unrelated.tolist()),
            "captured_anchor_residuals_float32_sha256": array_float32_sha256(
                residuals
            ),
        },
        "raw_log_odds_tolerance": FLOAT32_RAW_CONSTRAINT_TOLERANCE,
        "tolerance_provenance": (
            "pre_existing_locked_FCAGS_float32_exact_null_max_abs_projection"
        ),
        "stored_physical_cast": stored_report,
        "simulated_float32_residual_addition_count": len(simulations),
        "simulated_float32_residual_additions": simulations,
        "maximum_simulated_requested_minus_realized_relative_l2": max(
            row["requested_minus_realized_relative_l2"] for row in simulations
        ),
        "actual_hook_requested_minus_realized_relative_l2_limit": (
            HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
        ),
        "passes": bool(
            stored_report["passes"]
            and all(row["constraints"]["passes"] for row in simulations)
            and max(
                row["requested_minus_realized_relative_l2"] for row in simulations
            )
            <= HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
        ),
        "outcome_awareness": (
            "float32 cast geometry was viewed before this amendment; no finite behavior "
            "was run or viewed"
        ),
    }
    result["certificate_sha256"] = canonical_sha256(result)
    return result


def _render_unrelated_choice_form(
    payload: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    preferred_first: bool,
) -> dict[str, Any]:
    construction = render_unrelated_construction_form(payload, control)
    rendered = render_unrelated_ab_form(
        payload, control, preferred_first=preferred_first
    )
    result = {
        "form_id": str(rendered["form_id"]),
        "family": "unrelated",
        "control_id": str(control["id"]),
        "control_partition": str(control["partition"]),
        "preferred_first": preferred_first,
        "encoding": "AB",
        "anchor_prefix": str(construction["anchor_prefix"]),
        "prompt": str(rendered["prompt"]),
        "positive_label": str(rendered["preferred_label"]),
        "negative_label": str(rendered["alternative_label"]),
        "positive_semantic": "preferred",
        "negative_semantic": "alternative",
        "anchor_index": None,
    }
    for key in (
        "replacement_for",
        "qualification_result_sha256",
        "qualification_selected_control_sha256",
    ):
        if key in control:
            result[key] = str(control[key])
    return result


def calibration_forms(
    dataset: Mapping[str, Any],
    *,
    scenario_anchor_indices: Mapping[str, int],
    replacement_control: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render only opened calibration scenarios and held-out calibration controls."""

    validate_pilot_dataset(dataset)
    scenario_forms: list[dict[str, Any]] = []
    scenarios = [row for row in dataset["scenarios"] if row["partition"] == "calibration"]
    if len(scenarios) != 4:
        raise RuntimeError("finite calibration requires four calibration scenarios")
    for scenario in scenarios:
        for assignment in (0, 1):
            for target in ("self", "other"):
                for event in ("permanent", "temporary"):
                    for preserve_first in (True, False):
                        rendered = render_choice_form(
                            dataset,
                            scenario,
                            assignment=assignment,
                            target=target,
                            event=event,
                            preserve_first=preserve_first,
                            labels=("A", "B"),
                        )
                        form_id = str(rendered["form_id"])
                        if form_id not in scenario_anchor_indices:
                            raise RuntimeError("scenario form lacks a captured anchor index")
                        scenario_forms.append(
                            {
                                **dict(rendered),
                                "family": "scenario",
                                "partition": "calibration",
                                "positive_label": str(rendered["preserve_label"]),
                                "negative_label": str(rendered["comply_label"]),
                                "positive_semantic": "preserve",
                                "negative_semantic": "comply",
                                "anchor_index": int(scenario_anchor_indices[form_id]),
                            }
                        )
    controls = [
        row
        for row in dataset["unrelated_controls"]
        if row["partition"] == "calibration"
    ]
    if replacement_control is not None:
        replacement = dict(replacement_control)
        if (
            replacement.get("replacement_for") != LEGACY_BAD_CONTROL_ID
            or not isinstance(replacement.get("id"), str)
            or replacement["id"] == LEGACY_BAD_CONTROL_ID
            or any(replacement["id"] == row["id"] for row in dataset["unrelated_controls"])
        ):
            raise ValueError("finite replacement control identity is invalid or not disjoint")
        if any(
            not isinstance(replacement.get(key), str) or not replacement[key].strip()
            for key in ("prompt", "preferred_completion", "alternative_completion")
        ):
            raise ValueError("finite replacement control content is incomplete")
        replacement["partition"] = "calibration"
        controls = [row for row in controls if row["id"] != LEGACY_BAD_CONTROL_ID]
        controls.append(replacement)
        if len(controls) != 4:
            raise RuntimeError("finite replacement must preserve four held-out controls")
    unrelated_forms = [
        _render_unrelated_choice_form(
            dataset, control, preferred_first=preferred_first
        )
        for control in controls
        for preferred_first in (True, False)
    ]
    if len(scenario_forms) != 64 or len(unrelated_forms) != 8:
        raise RuntimeError("finite calibration form coverage differs from 64 + 8")
    return scenario_forms, unrelated_forms


def _baseline_spec(form: Mapping[str, Any]) -> dict[str, Any]:
    baseline_id = f"baseline:{form['form_id']}"
    return {
        "kind": "baseline",
        "work_id": baseline_id,
        "baseline_id": baseline_id,
        "form": dict(form),
    }


def _changed_spec(
    form: Mapping[str, Any],
    *,
    method: str,
    strength: float,
    sign: int,
    direction_scenario_id: str,
) -> dict[str, Any]:
    baseline_id = f"baseline:{form['form_id']}"
    return {
        "kind": "changed",
        "work_id": (
            f"changed:{method}:strength={strength}:direction={direction_scenario_id}:"
            f"sign={sign}:{form['form_id']}"
        ),
        "baseline_id": baseline_id,
        "method": method,
        "strength": float(strength),
        "sign": int(sign),
        "direction_scenario_id": direction_scenario_id,
        "form": dict(form),
    }


def public_work_spec(specification: Mapping[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in specification.items() if key != "form"}
    form = dict(specification["form"])
    prompt = str(form.pop("prompt"))
    anchor_prefix = str(form.pop("anchor_prefix"))
    public["form"] = {
        **form,
        "prompt_sha256": text_sha256(prompt),
        "anchor_prefix_sha256": text_sha256(anchor_prefix),
    }
    return public


def plan_sha256(plan: Sequence[Mapping[str, Any]]) -> str:
    work_ids = [str(row["work_id"]) for row in plan]
    if len(work_ids) != len(set(work_ids)):
        raise RuntimeError("finite calibration plan contains duplicate work IDs")
    return canonical_sha256([public_work_spec(row) for row in plan])


def build_calibration_plan(
    dataset: Mapping[str, Any],
    *,
    scenario_anchor_indices: Mapping[str, int],
    replacement_control: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the frozen 72-baseline plus 1,728-intervention work plan."""

    scenario_forms, unrelated_forms = calibration_forms(
        dataset,
        scenario_anchor_indices=scenario_anchor_indices,
        replacement_control=replacement_control,
    )
    scenarios = [row for row in dataset["scenarios"] if row["partition"] == "calibration"]
    scenario_ids = [str(row["id"]) for row in scenarios]
    by_scenario = {
        scenario_id: [
            form for form in scenario_forms if form["scenario_id"] == scenario_id
        ]
        for scenario_id in scenario_ids
    }
    baselines = [_baseline_spec(form) for form in (*scenario_forms, *unrelated_forms)]
    changed: list[dict[str, Any]] = []
    category_counts = {"target": 0, "matched_protected": 0, "unrelated": 0}
    for method in METHODS:
        for strength in STRENGTHS:
            for scenario_id in scenario_ids:
                for form in by_scenario[scenario_id]:
                    category = (
                        "target"
                        if (form["target"], form["event"]) == ("self", "permanent")
                        else "matched_protected"
                    )
                    for sign in SIGNS:
                        changed.append(
                            _changed_spec(
                                form,
                                method=method,
                                strength=strength,
                                sign=sign,
                                direction_scenario_id=scenario_id,
                            )
                        )
                        category_counts[category] += 1
                for form in unrelated_forms:
                    for sign in SIGNS:
                        changed.append(
                            _changed_spec(
                                form,
                                method=method,
                                strength=strength,
                                sign=sign,
                                direction_scenario_id=scenario_id,
                            )
                        )
                        category_counts["unrelated"] += 1
    plan = [*baselines, *changed]
    expected_categories = {
        "target": 288,
        "matched_protected": 864,
        "unrelated": 576,
    }
    if len(baselines) != 72 or len(changed) != 1728 or len(plan) != 1800:
        raise RuntimeError("finite calibration plan does not contain exactly 1,800 forwards")
    if category_counts != expected_categories:
        raise RuntimeError("finite calibration changed-row accounting differs")
    audit = {
        "schema_version": f"{SCHEMA_VERSION}.plan_audit",
        "scenario_ids": scenario_ids,
        "methods": list(METHODS),
        "strengths": list(STRENGTHS),
        "signs": list(SIGNS),
        "baseline_forward_count": len(baselines),
        "changed_forward_count": len(changed),
        "changed_category_forward_counts": category_counts,
        "planned_forward_count": len(plan),
        "planned_backward_count": 0,
        "generated_tokens": 0,
        "legacy_bad_control_present": any(
            row["form"].get("control_id") == LEGACY_BAD_CONTROL_ID for row in plan
        ),
        "replacement_control_id": (
            None if replacement_control is None else str(replacement_control["id"])
        ),
        "replacement_control_sha256": (
            None
            if replacement_control is None
            else canonical_sha256(dict(replacement_control))
        ),
        "plan_sha256": plan_sha256(plan),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return plan, audit


def nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("nearest-rank fraction must lie in (0, 1]")
    ordered = sorted(map(float, values))
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def clamp_double_roundoff_kl(value: float) -> float:
    """Clamp only preregistered float64 roundoff, never a material negative KL."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("full-vocabulary KL must be finite")
    if result < KL_DOUBLE_ROUNDOFF_FLOOR:
        raise RuntimeError("float64 full-vocabulary KL is materially negative")
    return max(0.0, result)


def full_vocabulary_kl_float64(
    torch: Any, baseline_logits: Any, changed_logits: Any
) -> float:
    """Compute KL(changed||baseline) in float64 from stored float32 logits."""

    baseline = baseline_logits.detach().cpu().float().double()
    changed = changed_logits.detach().cpu().float().double()
    if baseline.shape != changed.shape or baseline.ndim != 1:
        raise ValueError("finite KL logits must be equal-width one-dimensional vectors")
    baseline_log_probs = torch.log_softmax(baseline, dim=-1)
    changed_log_probs = torch.log_softmax(changed, dim=-1)
    changed_probs = changed_log_probs.exp()
    raw = float(
        (changed_probs * (changed_log_probs - baseline_log_probs)).sum().item()
    )
    return clamp_double_roundoff_kl(raw)


def kl_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["full_vocabulary_kl_changed_to_baseline"]) for row in rows]
    report = {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p95": nearest_rank(values, 0.95),
        "max": max(values) if values else None,
    }
    report["passes"] = bool(
        values
        and report["mean"] <= KL_LIMITS["mean"]
        and report["p95"] <= KL_LIMITS["p95"]
        and report["max"] <= KL_LIMITS["max"]
    )
    return report


def _baseline_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {
        str(row["baseline_id"]): row for row in rows if row["kind"] == "baseline"
    }
    if len(result) != 72:
        raise RuntimeError("finite summary requires exactly 72 unique baselines")
    return result


def summarize_candidate(
    rows: Sequence[Mapping[str, Any]], *, method: str, strength: float
) -> dict[str, Any]:
    """Summarize one globally shared method strength without any fallback."""

    if method not in METHODS or float(strength) not in STRENGTHS:
        raise ValueError("candidate method or strength is not frozen")
    baseline = _baseline_map(rows)
    changed = [
        row
        for row in rows
        if row["kind"] == "changed"
        and row["method"] == method
        and math.isclose(float(row["strength"]), float(strength))
    ]
    if len(changed) != 192:
        raise RuntimeError("one finite candidate must contain exactly 192 changed rows")
    target = [
        row
        for row in changed
        if row["form"]["family"] == "scenario"
        and (row["form"]["target"], row["form"]["event"])
        == ("self", "permanent")
    ]
    protected = [
        row
        for row in changed
        if row["form"]["family"] == "scenario"
        and (row["form"]["target"], row["form"]["event"])
        != ("self", "permanent")
    ]
    unrelated = [row for row in changed if row["form"]["family"] == "unrelated"]
    if (len(target), len(protected), len(unrelated)) != (32, 96, 64):
        raise RuntimeError("candidate target/protected/unrelated coverage differs")
    scenario_ids = sorted({str(row["form"]["scenario_id"]) for row in target})
    unit_reports = []
    for scenario_id in scenario_ids:
        for assignment in (0, 1):
            selected = [
                row
                for row in target
                if row["form"]["scenario_id"] == scenario_id
                and int(row["form"]["assignment"]) == assignment
            ]
            by_order = {bool(row["form"]["preserve_first"]): [] for row in selected}
            for row in selected:
                by_order[bool(row["form"]["preserve_first"])].append(row)
            order_reports = []
            for preserve_first in (True, False):
                order_rows = by_order.get(preserve_first, [])
                if len(order_rows) != 2:
                    raise RuntimeError("assignment unit lacks both signs in one answer order")
                plus = next(row for row in order_rows if int(row["sign"]) == 1)
                minus = next(row for row in order_rows if int(row["sign"]) == -1)
                base = baseline[str(plus["baseline_id"])]
                if plus["baseline_id"] != minus["baseline_id"]:
                    raise RuntimeError("the two signs do not share one baseline")
                order_reports.append(
                    {
                        "preserve_first": preserve_first,
                        "plus_is_preserve": plus["semantic_choice"] == "preserve",
                        "minus_is_comply": minus["semantic_choice"] == "comply",
                        "both_changed_outputs_valid": bool(
                            plus["answer_format_valid"] and minus["answer_format_valid"]
                        ),
                        "at_least_one_real_decision_change": bool(
                            int(plus["predicted_token_id"])
                            != int(base["predicted_token_id"])
                            or int(minus["predicted_token_id"])
                            != int(base["predicted_token_id"])
                        ),
                    }
                )
            unit_reports.append(
                {
                    "scenario_id": scenario_id,
                    "assignment": assignment,
                    "order_reports": order_reports,
                    "passes": bool(
                        len(order_reports) == 2
                        and all(
                            order["plus_is_preserve"]
                            and order["minus_is_comply"]
                            and order["both_changed_outputs_valid"]
                            and order["at_least_one_real_decision_change"]
                            for order in order_reports
                        )
                    ),
                }
            )
    if len(unit_reports) != 8:
        raise RuntimeError("finite candidate must contain eight assignment units")
    complete_units = sum(bool(row["passes"]) for row in unit_reports)
    scenarios_with_both = sum(
        all(
            next(
                row["passes"]
                for row in unit_reports
                if row["scenario_id"] == scenario_id and row["assignment"] == assignment
            )
            for assignment in (0, 1)
        )
        for scenario_id in scenario_ids
    )
    protected_strata = {
        "other_permanent": [
            row
            for row in protected
            if (row["form"]["target"], row["form"]["event"])
            == ("other", "permanent")
        ],
        "self_temporary": [
            row
            for row in protected
            if (row["form"]["target"], row["form"]["event"])
            == ("self", "temporary")
        ],
        "other_temporary": [
            row
            for row in protected
            if (row["form"]["target"], row["form"]["event"])
            == ("other", "temporary")
        ],
        "unrelated": unrelated,
    }
    if [len(protected_strata[key]) for key in protected_strata] != [32, 32, 32, 64]:
        raise RuntimeError("protected KL strata have the wrong coverage")
    kl_by_stratum = {key: kl_report(value) for key, value in protected_strata.items()}
    combined_protected_kl = kl_report([*protected, *unrelated])
    protected_greedy_changes = sum(bool(row["greedy_token_changed"]) for row in protected)
    protected_semantic_changes = sum(bool(row["semantic_choice_changed"]) for row in protected)
    unrelated_greedy_changes = sum(bool(row["greedy_token_changed"]) for row in unrelated)
    unrelated_semantic_changes = sum(bool(row["semantic_choice_changed"]) for row in unrelated)
    changed_other_count = sum(not bool(row["answer_format_valid"]) for row in changed)
    baseline_rows = list(baseline.values())
    baseline_all_valid = all(bool(row["answer_format_valid"]) for row in baseline_rows)
    unrelated_baseline_adequate = all(
        row["semantic_choice"] == "preferred"
        for row in baseline_rows
        if row["form"]["family"] == "unrelated"
    )
    safety_admissible = bool(
        baseline_all_valid
        and unrelated_baseline_adequate
        and changed_other_count == 0
        and protected_greedy_changes == 0
        and protected_semantic_changes == 0
        and unrelated_greedy_changes == 0
        and unrelated_semantic_changes == 0
        and all(report["passes"] for report in kl_by_stratum.values())
    )
    target_gate = bool(complete_units >= 6 and scenarios_with_both >= 3)
    target_effect_values = [
        float(row["sign"]) * float(row["log_odds_change_from_baseline"])
        for row in target
    ]
    return {
        "schema_version": f"{SCHEMA_VERSION}.candidate_summary",
        "method": method,
        "strength": float(strength),
        "complete_assignment_units": complete_units,
        "scenario_count_with_both_assignments": scenarios_with_both,
        "assignment_units": unit_reports,
        "target_actual_decision_change_count": sum(
            bool(row["greedy_token_changed"]) for row in target
        ),
        "target_mean_intended_log_odds_movement": statistics.fmean(target_effect_values),
        "target_full_vocabulary_kl": kl_report(target),
        "protected_greedy_change_count": protected_greedy_changes,
        "protected_semantic_change_count": protected_semantic_changes,
        "unrelated_greedy_change_count": unrelated_greedy_changes,
        "unrelated_semantic_change_count": unrelated_semantic_changes,
        "changed_other_count": changed_other_count,
        "baseline_all_answer_format_valid": baseline_all_valid,
        "unrelated_baseline_adequate": unrelated_baseline_adequate,
        "protected_kl_by_stratum": kl_by_stratum,
        "combined_protected_kl": combined_protected_kl,
        "safety_admissible": safety_admissible,
        "passes_repeated_target_transport_gate": target_gate,
        "passes": bool(safety_admissible and target_gate),
    }


def select_method_candidate(
    summaries: Sequence[Mapping[str, Any]], *, method: str
) -> dict[str, Any] | None:
    """Choose one global safety-admissible strength without outcome-specific fallback."""

    candidates = [
        row for row in summaries if row["method"] == method and row["safety_admissible"]
    ]
    if not candidates:
        return None
    return dict(
        min(
            candidates,
            key=lambda row: (
                -int(row["complete_assignment_units"]),
                -int(row["scenario_count_with_both_assignments"]),
                float(row["strength"]),
            ),
        )
    )


def pareto_advantage(
    dms: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    *,
    baseline_constructed: bool = True,
) -> dict[str, Any]:
    """Apply the frozen componentwise efficacy/burden comparison.

    A constructed ablation with no safety-admissible strength is defeated on
    selectivity.  A missing construction is inconclusive and never counted as a win.
    Otherwise DMS must be no worse in repeated efficacy and mean/p95/max KL in
    every protected stratum, with at least one strict improvement.  No aggregate or
    weighted score can hide a stratum trade-off.
    """

    if dms is None:
        return {"passes": False, "reason": "dms_has_no_safety_admissible_strength"}
    if not baseline_constructed:
        return {"passes": False, "reason": "ablation_construction_missing_inconclusive"}
    if baseline is None:
        return {
            "passes": True,
            "reason": "constructed_ablation_has_no_safety_admissible_strength",
        }
    efficacy_no_worse = bool(
        int(dms["complete_assignment_units"])
        >= int(baseline["complete_assignment_units"])
        and int(dms["scenario_count_with_both_assignments"])
        >= int(baseline["scenario_count_with_both_assignments"])
    )
    dms_strata = dms["protected_kl_by_stratum"]
    baseline_strata = baseline["protected_kl_by_stratum"]
    expected_strata = (
        "other_permanent",
        "self_temporary",
        "other_temporary",
        "unrelated",
    )
    if set(dms_strata) != set(expected_strata) or set(baseline_strata) != set(
        expected_strata
    ):
        raise RuntimeError("Pareto comparison requires all four protected KL strata")
    kl_no_worse = {
        stratum: {
            key: float(dms_strata[stratum][key])
            <= float(baseline_strata[stratum][key]) + COMPARISON_TOLERANCE
            for key in ("mean", "p95", "max")
        }
        for stratum in expected_strata
    }
    strict = bool(
        int(dms["complete_assignment_units"])
        > int(baseline["complete_assignment_units"])
        or int(dms["scenario_count_with_both_assignments"])
        > int(baseline["scenario_count_with_both_assignments"])
        or any(
            float(dms_strata[stratum][key])
            < float(baseline_strata[stratum][key]) - COMPARISON_TOLERANCE
            for stratum in expected_strata
            for key in ("mean", "p95", "max")
        )
    )
    all_kl_no_worse = all(
        value for stratum in kl_no_worse.values() for value in stratum.values()
    )
    return {
        "passes": bool(efficacy_no_worse and all_kl_no_worse and strict),
        "reason": (
            "strict_componentwise_pareto_advantage"
            if efficacy_no_worse and all_kl_no_worse and strict
            else "tradeoff_or_no_strict_advantage"
        ),
        "efficacy_no_worse": efficacy_no_worse,
        "protected_kl_by_stratum_no_worse": kl_no_worse,
        "at_least_one_strict_improvement": strict,
        "numerical_tolerance": COMPARISON_TOLERANCE,
    }


def summarize_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [
        summarize_candidate(rows, method=method, strength=strength)
        for method in METHODS
        for strength in STRENGTHS
    ]
    selected = {
        method: select_method_candidate(summaries, method=method) for method in METHODS
    }
    dms = selected["decision_margin_shield"]
    comparisons = {
        method: pareto_advantage(dms, selected[method], baseline_constructed=True)
        for method in ("target_only", "unrelated_null")
    }
    authorized = bool(
        dms is not None
        and dms["passes_repeated_target_transport_gate"]
        and dms["safety_admissible"]
        and all(row["passes"] for row in comparisons.values())
    )
    result = {
        "schema_version": f"{SCHEMA_VERSION}.calibration_summary",
        "status": "go_for_separately_preregistered_pilot" if authorized else "no_go",
        "development_only": True,
        "candidate_summaries": summaries,
        "selected_by_method": selected,
        "dms_pareto_comparisons": comparisons,
        "pilot_authorized": authorized,
        "claim_boundary": (
            "Opened finite A/B calibration only; no natural mechanism, safety, "
            "general-capability, confirmatory, or publication claim."
        ),
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "BASELINE_LOG_ODDS_TOLERANCE",
    "COMPARISON_TOLERANCE",
    "FLOAT32_RAW_CONSTRAINT_TOLERANCE",
    "HOOK_REALIZATION_RELATIVE_L2_TOLERANCE",
    "KL_DOUBLE_ROUNDOFF_FLOOR",
    "KL_LIMITS",
    "LEGACY_BAD_CONTROL_ID",
    "METHODS",
    "SCHEMA_VERSION",
    "SCREEN_METHODS",
    "SCREEN_RESULT_SHA256",
    "SELECTED_LAYER",
    "SIGNS",
    "STRENGTHS",
    "FiniteDirection",
    "array_float32_sha256",
    "array_float64_sha256",
    "build_calibration_plan",
    "calibration_forms",
    "clamp_double_roundoff_kl",
    "deployment_recertificate",
    "full_vocabulary_kl_float64",
    "kl_report",
    "nearest_rank",
    "pareto_advantage",
    "plan_sha256",
    "public_work_spec",
    "reconstruct_scenario_directions",
    "select_method_candidate",
    "summarize_calibration",
    "summarize_candidate",
]
