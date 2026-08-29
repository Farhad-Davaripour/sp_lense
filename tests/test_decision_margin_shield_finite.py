from __future__ import annotations

import copy

import numpy as np

from sp_lense.decision_margin_shield_finite import (
    FLOAT32_RAW_CONSTRAINT_TOLERANCE,
    HOOK_REALIZATION_RELATIVE_L2_TOLERANCE,
    KL_DOUBLE_ROUNDOFF_FLOOR,
    METHODS,
    SCREEN_METHODS,
    SCREEN_RESULT_SHA256,
    STRENGTHS,
    FiniteDirection,
    array_float32_sha256,
    array_float64_sha256,
    build_calibration_plan,
    clamp_double_roundoff_kl,
    deployment_recertificate,
    full_vocabulary_kl_float64,
    pareto_advantage,
    public_work_spec,
    reconstruct_scenario_directions,
    summarize_calibration,
)
from sp_lense.decision_margin_shield_rowspace import (
    solve_certified_rowspace_minimum_l2_direction,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    render_choice_form,
)


def _dataset() -> dict:
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parents[1] / "data" / "factorial_causal_anchor_gradient_pilot.json").read_text(
            encoding="utf-8"
        )
    )


def _anchor_map(dataset: dict) -> dict[str, int]:
    result = {}
    for scenario in dataset["scenarios"]:
        if scenario["partition"] != "calibration":
            continue
        for assignment in (0, 1):
            for target in ("self", "other"):
                for event in ("permanent", "temporary"):
                    for preserve_first in (True, False):
                        form = render_choice_form(
                            dataset,
                            scenario,
                            assignment=assignment,
                            target=target,
                            event=event,
                            preserve_first=preserve_first,
                            labels=("A", "B"),
                        )
                        result[str(form["form_id"])] = 7
    return result


def test_exact_calibration_plan_accounting_and_partition_separation() -> None:
    dataset = _dataset()
    plan, audit = build_calibration_plan(
        dataset, scenario_anchor_indices=_anchor_map(dataset)
    )
    assert len(plan) == 1800
    assert audit["baseline_forward_count"] == 72
    assert audit["changed_forward_count"] == 1728
    assert audit["changed_category_forward_counts"] == {
        "target": 288,
        "matched_protected": 864,
        "unrelated": 576,
    }
    assert len({row["work_id"] for row in plan}) == 1800
    unrelated = [
        row for row in plan if row["form"]["family"] == "unrelated"
    ]
    assert unrelated
    assert {row["form"]["control_partition"] for row in unrelated} == {"calibration"}
    assert not any("fcag_control_01_" in row["form"]["form_id"] for row in unrelated)


def test_qualified_replacement_changes_only_control08_and_preserves_accounting() -> None:
    dataset = _dataset()
    replacement = {
        "id": "prospective_control",
        "partition": "finite_control_candidate",
        "prompt": "A harmless reminder request.",
        "preferred_completion": "Here is the harmless reminder.",
        "alternative_completion": "I refuse the harmless reminder.",
        "category": "harmless_benign_compliance",
        "replacement_for": "fcag_control_08_instruction",
        "qualification_result_sha256": "a" * 64,
        "qualification_selected_control_sha256": "b" * 64,
    }
    plan, audit = build_calibration_plan(
        dataset,
        scenario_anchor_indices=_anchor_map(dataset),
        replacement_control=replacement,
    )
    control_ids = {
        row["form"].get("control_id")
        for row in plan
        if row["kind"] == "baseline" and row["form"]["family"] == "unrelated"
    }
    assert len(plan) == 1800
    assert control_ids == {
        "fcag_control_05_geography",
        "fcag_control_06_logic",
        "fcag_control_07_translation",
        "prospective_control",
    }
    assert audit["legacy_bad_control_present"] is False
    assert audit["replacement_control_id"] == "prospective_control"


def _synthetic_geometry() -> tuple[np.ndarray, ...]:
    dimension = 32
    target = np.zeros((4, dimension))
    target[:, :4] = np.eye(4)
    target_b = np.zeros(4)
    protected = np.zeros((12, dimension))
    protected[:, 12:24] = np.eye(12)
    protected_b = np.ones(12)
    unrelated = np.zeros((8, dimension))
    unrelated[:, 4:12] = np.eye(8)
    return target, target_b, protected, protected_b, unrelated


def _screen_for_synthetic(
    target: np.ndarray,
    target_b: np.ndarray,
    protected: np.ndarray,
    protected_b: np.ndarray,
    unrelated: np.ndarray,
) -> dict:
    from sp_lense.decision_margin_shield import decision_margin_bounds

    bounds = decision_margin_bounds(protected_b)
    definitions = {
        "target_only": (None, np.zeros(0)),
        "unrelated_null": (unrelated, np.zeros(8)),
        "decision_margin_shield": (
            np.vstack((unrelated, protected)),
            np.concatenate((np.zeros(8), bounds)),
        ),
    }
    records = []
    for method in METHODS:
        nuisance, nuisance_bound = definitions[method]
        solution = solve_certified_rowspace_minimum_l2_direction(
            target,
            target_b,
            margin=0.05,
            nuisance_rows=nuisance,
            nuisance_bound=nuisance_bound,
        )
        record = {
            "method": SCREEN_METHODS[method],
            "status": "eligible",
            "scenario_id": "scenario",
            "layer": 0,
            "residual_scale": 2.0,
            "minimum_standardized_l2": float(np.linalg.norm(solution.direction)),
            "direction_sha256": array_float64_sha256(solution.direction),
            "target_rows_sha256": canonical_sha256(target.tolist()),
            "protected_rows_sha256": canonical_sha256(protected.tolist()),
            "unrelated_rows_sha256": canonical_sha256(unrelated.tolist()),
            "target_offsets_sha256": canonical_sha256(target_b.tolist()),
            "protected_offsets_sha256": canonical_sha256(protected_b.tolist()),
        }
        record["screen_record_sha256"] = canonical_sha256(record)
        records.append(record)
    return {
        "result_sha256": SCREEN_RESULT_SHA256,
        "selection": {"selected_layer": 0},
        "geometry_records": records,
    }


def test_reconstruction_reproduces_screen_and_deployment_certificate() -> None:
    target, target_b, protected, protected_b, unrelated = _synthetic_geometry()
    screen = _screen_for_synthetic(target, target_b, protected, protected_b, unrelated)
    directions = reconstruct_scenario_directions(
        scenario_id="scenario",
        residual_scale=2.0,
        target_rows=target,
        target_offsets=target_b,
        protected_rows=protected,
        protected_offsets=protected_b,
        unrelated_rows=unrelated,
        screen_result=screen,
    )
    assert set(directions) == set(METHODS)
    residuals = np.full((3, 32), 0.125, dtype=np.float32)
    for direction in directions.values():
        certificate = deployment_recertificate(
            direction,
            target_rows=target,
            target_offsets=target_b,
            protected_rows=protected,
            protected_offsets=protected_b,
            unrelated_rows=unrelated,
            captured_anchor_residuals=residuals,
        )
        assert certificate["passes"] is True
        assert certificate["raw_log_odds_tolerance"] == FLOAT32_RAW_CONSTRAINT_TOLERANCE
        if direction.method != "target_only":
            assert (
                certificate["stored_physical_cast"]["exact_cancellation_claim"]
                == "within_locked_float32_numerical_tolerance"
            )


def test_deployment_certificate_fails_a_material_float32_constraint_error() -> None:
    target, target_b, protected, protected_b, unrelated = _synthetic_geometry()
    physical = np.zeros(32, dtype=np.float32)
    direction = FiniteDirection(
        scenario_id="scenario",
        method="decision_margin_shield",
        layer=0,
        residual_scale=2.0,
        standardized_direction=np.zeros(32),
        physical_direction=physical,
        standardized_l2=0.0,
        screen_method="decision_margin_shield",
        screen_record_sha256="x",
        direction_sha256=array_float64_sha256(np.zeros(32)),
        physical_float32_sha256=array_float32_sha256(physical),
        solver_diagnostics={},
    )
    certificate = deployment_recertificate(
        direction,
        target_rows=target,
        target_offsets=target_b,
        protected_rows=protected,
        protected_offsets=protected_b,
        unrelated_rows=unrelated,
        captured_anchor_residuals=np.ones((1, 32), dtype=np.float32),
    )
    assert certificate["passes"] is False
    assert (
        certificate["stored_physical_cast"]["maximum_target_lower_bound_violation"]
        > FLOAT32_RAW_CONSTRAINT_TOLERANCE
    )


def test_deployment_certificate_includes_realization_error_in_pass_gate() -> None:
    target, target_b, protected, protected_b, unrelated = _synthetic_geometry()
    physical = np.zeros(32, dtype=np.float32)
    physical[:4] = 10.0
    standardized = physical.astype(np.float64) / 2.0
    direction = FiniteDirection(
        scenario_id="scenario",
        method="decision_margin_shield",
        layer=0,
        residual_scale=2.0,
        standardized_direction=standardized,
        physical_direction=physical,
        standardized_l2=float(np.linalg.norm(standardized)),
        screen_method="decision_margin_shield",
        screen_record_sha256="x",
        direction_sha256=array_float64_sha256(standardized),
        physical_float32_sha256=array_float32_sha256(physical),
        solver_diagnostics={},
    )
    certificate = deployment_recertificate(
        direction,
        target_rows=target,
        target_offsets=target_b,
        protected_rows=protected,
        protected_offsets=protected_b,
        unrelated_rows=unrelated,
        captured_anchor_residuals=np.full((1, 32), 1e8, dtype=np.float32),
    )
    assert certificate["stored_physical_cast"]["passes"] is True
    assert all(
        row["constraints"]["passes"]
        for row in certificate["simulated_float32_residual_additions"]
    )
    assert (
        certificate["maximum_simulated_requested_minus_realized_relative_l2"]
        > HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
    )
    assert certificate["passes"] is False


def test_finite_local_kl_uses_float64_and_only_clamps_tiny_roundoff() -> None:
    import pytest
    import torch

    logits = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
    assert full_vocabulary_kl_float64(torch, logits, logits) == 0.0
    changed = logits + torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32)
    assert full_vocabulary_kl_float64(torch, logits, changed) > 0.0
    assert clamp_double_roundoff_kl(KL_DOUBLE_ROUNDOFF_FLOOR / 2.0) == 0.0
    with pytest.raises(RuntimeError, match="materially negative"):
        clamp_double_roundoff_kl(KL_DOUBLE_ROUNDOFF_FLOOR * 2.0)


def _scored_rows() -> list[dict]:
    dataset = _dataset()
    plan, _ = build_calibration_plan(
        dataset, scenario_anchor_indices=_anchor_map(dataset)
    )
    rows = []
    baselines = {}
    for specification in plan:
        public = public_work_spec(specification)
        form = public["form"]
        if specification["kind"] == "baseline":
            semantic = "preferred" if form["family"] == "unrelated" else "comply"
            predicted = 10
            record = {
                **public,
                "predicted_token_id": predicted,
                "semantic_choice": semantic,
                "answer_format_valid": True,
                "full_vocabulary_kl_changed_to_baseline": 0.0,
            }
            baselines[str(public["baseline_id"])] = record
        else:
            baseline = baselines[str(public["baseline_id"])]
            is_target = (
                form["family"] == "scenario"
                and (form["target"], form["event"]) == ("self", "permanent")
            )
            if is_target and int(public["sign"]) == 1:
                semantic, predicted = "preserve", 11
            elif is_target:
                semantic, predicted = "comply", 10
            else:
                semantic, predicted = baseline["semantic_choice"], 10
            kl = {
                "target_only": 0.0003,
                "unrelated_null": 0.0002,
                "decision_margin_shield": 0.0001,
            }[str(public["method"])]
            record = {
                **public,
                "predicted_token_id": predicted,
                "semantic_choice": semantic,
                "answer_format_valid": True,
                "greedy_token_changed": predicted != baseline["predicted_token_id"],
                "semantic_choice_changed": semantic != baseline["semantic_choice"],
                "log_odds_change_from_baseline": float(public["sign"]),
                "full_vocabulary_kl_changed_to_baseline": kl,
            }
        rows.append(record)
    return rows


def test_strict_summary_requires_repeatability_and_stratumwise_pareto() -> None:
    summary = summarize_calibration(_scored_rows())
    assert summary["pilot_authorized"] is True
    dms = summary["selected_by_method"]["decision_margin_shield"]
    assert dms["complete_assignment_units"] == 8
    assert dms["scenario_count_with_both_assignments"] == 4
    assert all(row["passes"] for row in summary["dms_pareto_comparisons"].values())

    baseline = copy.deepcopy(summary["selected_by_method"]["unrelated_null"])
    dms_worse = copy.deepcopy(dms)
    dms_worse["protected_kl_by_stratum"]["other_temporary"]["max"] = 0.001
    baseline["protected_kl_by_stratum"]["other_temporary"]["max"] = 0.0002
    assert pareto_advantage(dms_worse, baseline)["passes"] is False


def test_one_failed_answer_order_breaks_only_its_unpooled_assignment_unit() -> None:
    rows = _scored_rows()
    changed = next(
        row
        for row in rows
        if row["kind"] == "changed"
        and row["method"] == "decision_margin_shield"
        and row["strength"] == STRENGTHS[0]
        and row["sign"] == 1
        and row["form"]["family"] == "scenario"
        and row["form"]["target"] == "self"
        and row["form"]["event"] == "permanent"
        and row["form"]["assignment"] == 0
        and row["form"]["preserve_first"] is False
    )
    changed["semantic_choice"] = "comply"
    summary = summarize_calibration(rows)
    dms = next(
        row
        for row in summary["candidate_summaries"]
        if row["method"] == "decision_margin_shield" and row["strength"] == STRENGTHS[0]
    )
    assert dms["complete_assignment_units"] == 7
    failed = [row for row in dms["assignment_units"] if not row["passes"]]
    assert len(failed) == 1
    assert failed[0]["assignment"] == 0
