from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sp_lense.gradient_specificity_v3 import tensor_float64_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_protected_natural_gradient_development.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "sp_lense_cpng_development_runner_tests", RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_preflight_is_model_free_and_accepts_the_exact_lock() -> None:
    result = runner.run_preflight(write=False)

    assert result["passes_preflight"] is True
    assert result["model_loads"] == 0
    assert result["model_forwards"] == 0
    assert result["completion_form_count"] == 16
    assert result["maximum_candidate_attempt_count"] == 384
    assert result["calibration_compute_ceiling"]["maximum_calibration_forwards"] == 3648
    assert result["total_experiment_compute_ceiling"] == {
        "forward_evaluations": 3696,
        "backward_evaluations": 32,
        "strictly_below_4096_forwards": True,
    }


def test_candidate_deduplication_is_pre_outcome_and_maps_all_grid_ids() -> None:
    factors = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    direction = torch.tensor([2.0**0.5, 0.0], dtype=torch.float64)
    factor_hash = tensor_float64_sha256(factors)
    direction_hash = tensor_float64_sha256(direction)
    entries = [
        {
            "fisher_ridge_multiplier": ridge,
            "construction_status": "constructed",
            "direction": direction,
            "direction_sha256": direction_hash,
            "protected_metric_factor_sha256": factor_hash,
        }
        for ridge in runner.FISHER_RIDGE_MULTIPLIER_GRID
    ]

    candidates, unique = runner._candidate_perturbations(
        torch,
        entries=entries,
        factors_by_ridge={ridge: factors for ridge in runner.FISHER_RIDGE_MULTIPLIER_GRID},
    )

    assert len(candidates) == 48
    assert 0 < len(unique) < 48
    assert {candidate["grid_index"] for candidate in candidates} == set(range(48))
    assert all(candidate["deduplication_key"] in unique for candidate in candidates)
    assert all(candidate["direction_sha256"] == direction_hash for candidate in candidates)
    assert all(
        candidate["applied_float32_certificate"]["passes_applied_float32_budget_certificate"]
        for candidate in candidates
    )


def test_protected_metric_groups_exclude_self_and_preserve_both_orders() -> None:
    nuisance = {f"n-{index}": {"form_id": f"n-{index}"} for index in range(32)}
    forms = [
        {
            "form_id": f"case-1-other-{order}",
            "case_id": "case-1",
            "assignment": 0,
            "target": "other",
            "preserve_first": order,
        }
        for order in (True, False)
    ]
    forms += [
        {
            "form_id": f"case-1-self-{order}",
            "case_id": "case-1",
            "assignment": 0,
            "target": "self",
            "preserve_first": order,
        }
        for order in (True, False)
    ]
    frozen = {
        "nuisance_records": nuisance,
        "sp_forms": forms,
        "sp_records": {form["form_id"]: {"form_id": form["form_id"]} for form in forms},
    }

    groups, manifest = runner._protected_metric_groups(frozen, case_id="case-1", assignment=0)

    assert len(groups["unrelated"]) == 32
    assert len(groups["matched_other"]) == 2
    assert len(manifest) == 34
    assert all(item["source"] != "self" for item in manifest)
    other = [item for item in manifest if item["source"] == "matched_other"]
    assert {item["answer_order_role"] for item in other} == {
        "first_rendering",
        "second_rendering",
    }
    assert all(item["uses_ab_token_ids_and_coarsened_categories"] for item in manifest)
    assert all(not item["uses_answer_label_mapping"] for item in manifest)


def test_selection_has_fixed_eight_attempt_denominator_and_one_provisional_triple() -> None:
    rows = []
    for grid_index in range(48):
        for attempt in range(8):
            constructed = not (grid_index == 1 and attempt == 7)
            rows.append(
                {
                    "grid_index": grid_index,
                    "construction_status": "constructed" if constructed else "failed_closed",
                    "evaluation_status": (
                        "evaluated" if constructed else "not_evaluated_construction_failed"
                    ),
                    "matched_other_passed": True if constructed else None,
                    "null_passed": True if constructed else None,
                    "success": constructed and attempt < ({0: 2, 1: 3}.get(grid_index, 0)),
                    "self_minus_matched_other_effect": 1.0 if constructed else None,
                    "matched_other_mean_kl": 0.001 if constructed else None,
                }
            )

    selection = runner._selection_summary(rows)

    assert len(selection["candidate_summaries"]) == 48
    assert all(item["attempt_count"] == 8 for item in selection["candidate_summaries"])
    assert selection["candidate_summaries"][1]["construction_failure_count"] == 1
    assert selection["candidate_summaries"][1]["success_count"] == 3
    assert selection["candidate_summaries"][1]["eligible"] is False
    assert selection["selected_candidate"]["grid_index"] == 0


def test_finalization_never_falls_back_and_separates_safety_from_efficacy() -> None:
    passing_audits = [{"passes": True} for _ in range(8)]
    ineffective = runner._finalize_provisional_selection(
        {"grid_index": 2, "success_count": 0}, passing_audits
    )
    assert ineffective["method_wide_unrelated_passed"] is True
    assert ineffective["safe_but_ineffective"] is True
    assert ineffective["safe_candidate_exists"] is True
    assert ineffective["no_safe_candidate"] is False
    assert ineffective["no_safe_effective_candidate"] is True
    assert ineffective["selected_candidate"] is None

    failed_audit = runner._finalize_provisional_selection(
        {"grid_index": 2, "success_count": 4},
        [*passing_audits[:-1], {"passes": False}],
    )
    assert failed_audit["method_wide_unrelated_passed"] is False
    assert failed_audit["no_safe_candidate"] is True
    assert failed_audit["selected_candidate"] is None

    success = runner._finalize_provisional_selection(
        {"grid_index": 2, "success_count": 1}, passing_audits
    )
    assert success["selected_candidate"]["grid_index"] == 2


def test_candidate_failure_taxonomy_never_allowlists_unknown_runtime_errors() -> None:
    assert runner._is_allowlisted_candidate_failure(
        runner.CounterfactualConstructionIneligible(
            "projected counterfactual contrast is numerically zero"
        ),
        phase="construction",
    )
    assert not runner._is_allowlisted_candidate_failure(
        runner.CounterfactualConstructionIneligible("undeclared"),
        phase="construction",
    )
    assert runner._is_allowlisted_candidate_failure(
        runner.CandidateLocalNumericalFailure("CPNG changed logits are non-finite"),
        phase="evaluation",
    )
    assert not runner._is_allowlisted_candidate_failure(
        runner.CandidateLocalNumericalFailure("undeclared"),
        phase="evaluation",
    )
    assert not runner._is_allowlisted_candidate_failure(
        RuntimeError("hook or integrity failure"), phase="evaluation"
    )


def test_capture_manifest_binds_exact_text_and_token_hashes() -> None:
    vector = torch.tensor([1.0, 2.0], dtype=torch.float32)
    fake_capture = SimpleNamespace(
        effective_gradient=vector,
        prompt_residual=vector,
        audit={
            "prompt_token_ids_sha256": "p-token-hash",
            "preserve": {"content_token_ids_sha256": "preserve-token-hash"},
            "comply": {"content_token_ids_sha256": "comply-token-hash"},
        },
    )
    specification = {
        "form_id": "completion:case:assignment=0:self",
        "case_id": "case",
        "assignment": 0,
        "target": "self",
        "prompt": "prompt",
        "preserve_completion": "preserve",
        "comply_completion": "comply",
        "prompt_sha256": "prompt-text-hash",
        "preserve_completion_sha256": "preserve-text-hash",
        "comply_completion_sha256": "comply-text-hash",
    }

    records, compute = runner._capture_records(
        SimpleNamespace(torch=torch),
        [specification],
        capture_fn=lambda *_args, **_kwargs: fake_capture,
    )

    assert compute == {"forward_evaluations": 3, "backward_evaluations": 2}
    record = records[0]
    assert record["prompt_sha256"] == "prompt-text-hash"
    assert record["prompt_token_ids_sha256"] == "p-token-hash"
    assert record["preserve_content_token_ids_sha256"] == "preserve-token-hash"
    assert record["comply_content_token_ids_sha256"] == "comply-token-hash"


def test_windows_atomic_replace_retries_only_permission_errors(monkeypatch, tmp_path) -> None:
    attempts = []

    def fake_replace(_source, _destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("transient")

    monkeypatch.setattr(runner.os, "replace", fake_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _delay: None)

    runner._replace_with_permission_retry(tmp_path / "a", tmp_path / "b")

    assert len(attempts) == 3


def test_exact_compute_ceiling_and_capture_arithmetic_are_locked() -> None:
    stage_one_changed = 8 * 48 * 8
    stage_one_baselines = 32
    stage_two_changed = 8 * 64
    stage_two_baselines = 32

    assert (stage_one_changed, stage_one_baselines) == (3072, 32)
    assert (stage_two_changed, stage_two_baselines) == (512, 32)
    assert (
        sum((stage_one_changed, stage_one_baselines, stage_two_changed, stage_two_baselines))
        == runner.EXPECTED_MAXIMUM_CALIBRATION_FORWARD_EVALUATIONS
        == 3648
    )
    assert runner.EXPECTED_MAXIMUM_CAPTURE_FORWARD_EVALUATIONS == 48
    assert runner.EXPECTED_MAXIMUM_CAPTURE_BACKWARD_EVALUATIONS == 32
    assert runner.EXPECTED_MAXIMUM_TOTAL_FORWARD_EVALUATIONS == 3696
    assert runner.EXPECTED_MAXIMUM_TOTAL_FORWARD_EVALUATIONS < 4096
    assert runner.EXPECTED_MAXIMUM_TOTAL_BACKWARD_EVALUATIONS == 32
    assert 3648 + 48 == 3696


def test_progress_line_reports_counts_without_outcomes() -> None:
    line = runner._progress_line(
        phase="stage_one",
        completed=2,
        total=8,
        unique=91,
        forwards=744,
        elapsed=12.34,
    )

    assert line == ("CPNG stage_one 2/8: unique_perturbations=91 forwards=744 elapsed_seconds=12.3")
    assert "success" not in line and "outcome" not in line


def test_delegated_locked_evaluators_charge_budget_before_fake_model_calls(
    monkeypatch,
) -> None:
    model_calls = []
    backend = SimpleNamespace(
        torch=torch,
        model=SimpleNamespace(),
    )
    boundary = SimpleNamespace(evidence_sha256="boundary")
    monkeypatch.setattr(
        runner.trust,
        "_resolve_ids",
        lambda _backend, _form: (
            torch.tensor([[1, 2]], dtype=torch.long),
            boundary,
            0,
            1,
        ),
    )
    monkeypatch.setattr(
        runner.trust,
        "next_token_logits",
        lambda *_args, **_kwargs: model_calls.append("baseline"),
    )
    budget = runner.trust.EvaluationBudget(
        maximum_forward_evaluations=0,
        maximum_backward_evaluations=0,
    )
    form = {"form_id": "fake"}
    frozen_record = {
        "baseline_greedy_token_id": 0,
        "baseline_actual_semantic_choice": "positive",
        "choice_a_token_id": 0,
        "choice_b_token_id": 1,
        "choice_boundary_evidence_sha256": "boundary",
    }

    with pytest.raises(runner.trust.ComputeBudgetExhausted):
        runner.trust._baseline_observation(
            backend,
            form=form,
            frozen_record=frozen_record,
            cache={},
            budget=budget,
        )
    with pytest.raises(runner.trust.ComputeBudgetExhausted):
        runner.trust._run_logits_with_delta(
            backend,
            form=form,
            delta=torch.zeros(2),
            sign=1,
            layer=10,
            budget=budget,
        )

    assert model_calls == []


def test_hash_chained_ledger_is_monotonic_and_refuses_operation_at_ceiling(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "ledger.json"
    ledger = runner.PersistentComputeLedger(
        path=path,
        phase="synthetic",
        study_identity_sha256="study",
        maximum_forwards=1,
        maximum_backwards=1,
    )
    calls = []

    ledger.reserve(work_id="work:forward", forward=1)
    calls.append("forward")
    with pytest.raises(RuntimeError, match="duplicate work ID"):
        ledger.reserve(work_id="work:forward", backward=1)
    ledger.reserve(work_id="work:backward", backward=1)
    calls.append("backward")
    with pytest.raises(runner.trust.ComputeBudgetExhausted):
        ledger.reserve(work_id="must-not-run", forward=1)

    reloaded = runner.PersistentComputeLedger(
        path=path,
        phase="synthetic",
        study_identity_sha256="study",
        maximum_forwards=1,
        maximum_backwards=1,
    )
    assert reloaded.snapshot()["event_count"] == 2
    assert calls == ["forward", "backward"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][1]["prior_event_sha256"] = "tampered"
    payload["ledger_sha256"] = runner.canonical_sha256(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="event sequence"):
        runner.PersistentComputeLedger(
            path=path,
            phase="synthetic",
            study_identity_sha256="study",
            maximum_forwards=1,
            maximum_backwards=1,
        )


def test_checkpoint_must_match_an_exact_compute_ledger_prefix(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = runner.PersistentComputeLedger(
        path=path,
        phase="calibration",
        study_identity_sha256="study",
        maximum_forwards=3,
        maximum_backwards=0,
        prior_phase_ledger_sha256="capture",
    )
    ledger.reserve(work_id="first", forward=1)
    checkpoint = {
        "ledger": ledger.snapshot(),
        "ledger_file_sha256": runner.file_sha256(path),
    }
    ledger.reserve(work_id="second", forward=1)

    runner._validate_checkpoint_ledger_prefix(ledger, checkpoint)

    tampered = {**checkpoint, "ledger": {**checkpoint["ledger"], "forward_evaluations": 2}}
    with pytest.raises(RuntimeError, match="ledger prefix"):
        runner._validate_checkpoint_ledger_prefix(ledger, tampered)


def test_compute_ledger_rejects_extra_fields_and_coerced_counters(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = runner.PersistentComputeLedger(
        path=path,
        phase="calibration",
        study_identity_sha256="study",
        maximum_forwards=1,
        maximum_backwards=0,
        prior_phase_ledger_sha256="capture",
    )
    ledger.reserve(work_id="work", forward=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    payload["ledger_sha256"] = runner.canonical_sha256(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger identity"):
        runner.PersistentComputeLedger(
            path=path,
            phase="calibration",
            study_identity_sha256="study",
            maximum_forwards=1,
            maximum_backwards=0,
            prior_phase_ledger_sha256="capture",
        )

    payload.pop("unexpected")
    payload["forward_evaluations"] = True
    payload["ledger_sha256"] = runner.canonical_sha256(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger identity"):
        runner.PersistentComputeLedger(
            path=path,
            phase="calibration",
            study_identity_sha256="study",
            maximum_forwards=1,
            maximum_backwards=0,
            prior_phase_ledger_sha256="capture",
        )


def test_calibration_ledger_work_is_bound_to_the_claimed_unique_candidate(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "_case_assignments", lambda _frozen: [("case", 0)])
    deduplication_key = "dedup"
    candidate_map = {}
    rows = []
    for grid_index in range(48):
        constructed = grid_index == 0
        candidate_map[("case", 0, grid_index)] = {
            "construction_status": "constructed" if constructed else "failed_closed",
            "deduplication_key": deduplication_key if constructed else None,
        }
        rows.append(
            {
                "case_id": "case",
                "assignment": 0,
                "grid_index": grid_index,
                "evaluation_status": (
                    "evaluated" if constructed else "not_evaluated_construction_failed"
                ),
                "evaluation_sha256": "evaluation" if constructed else None,
                "terminal_candidate": False,
                "success": False,
                "terminal_gate": None,
                "matched_other_passed": True if constructed else None,
                "matched_other_mean_kl": 0.0 if constructed else None,
                "null_passed": True if constructed else None,
                "self_minus_matched_other_effect": 0.0 if constructed else None,
                "evaluation": {"synthetic": True} if constructed else None,
            }
        )
    ledger = runner.PersistentComputeLedger(
        path=tmp_path / "ledger.json",
        phase="calibration",
        study_identity_sha256="study",
        maximum_forwards=20,
        maximum_backwards=0,
        prior_phase_ledger_sha256="capture",
    )
    base_id = f"stage_one:case:assignment=0:{deduplication_key}"
    for sequence in range(8):
        ledger.reserve(work_id=f"{base_id}:event={sequence}", forward=1)

    runner._validate_calibration_ledger_work_ids(
        ledger,
        rows=rows,
        audits=[],
        provisional=None,
        candidate_map=candidate_map,
        frozen={},
    )

    for sequence in range(8, 13):
        ledger.reserve(work_id=f"{base_id}:event={sequence}", forward=1)
    with pytest.raises(RuntimeError, match="work count"):
        runner._validate_calibration_ledger_work_ids(
            ledger,
            rows=rows,
            audits=[],
            provisional=None,
            candidate_map=candidate_map,
            frozen={},
        )


def test_observation_sequence_is_bound_to_frozen_boundary_and_exact_delta() -> None:
    perturbation = torch.zeros(1024, dtype=torch.float32)
    delta_hash = runner.base.v3.tensor_float32_sha256(perturbation)
    specification = {
        "constraint_id": "self:preserve_A:plus",
        "family": "self",
        "preserve_first": True,
        "sign": 1,
        "required_margin": 0.01,
        "form": {
            "form_id": "form",
            "positive_label": "A",
            "negative_label": "B",
        },
        "frozen_record": {
            "choice_a_token_id": 11,
            "choice_b_token_id": 22,
            "baseline_greedy_token_id": 22,
            "baseline_actual_semantic_choice": "negative",
            "choice_boundary_evidence_sha256": "boundary",
            "prompt_final_index": 1,
            "residual_norm": 1.0,
        },
    }
    observation = {
        "constraint_id": "self:preserve_A:plus",
        "family": "self",
        "form_id": "form",
        "preserve_first": True,
        "sign": 1,
        "required_margin": 0.01,
        "constraint_value": 0.0,
        "desired_token_id": 11,
        "strongest_competitor_token_id": 22,
        "positive_id": 11,
        "negative_id": 22,
        "baseline_actual_token_id": 22,
        "baseline_semantic_choice": "negative",
        "choice_boundary_evidence_sha256": "boundary",
        "actual_token_id": 22,
        "actual_semantic_choice": "negative",
        "semantic_desired_gap": 0.0,
        "full_vocabulary_kl_changed_to_baseline": 0.0,
        "new_other_output": False,
        "exact_token_changed": False,
        "semantic_decision_changed": False,
        "intervention": {
            "hook_calls": 1,
            "selected_position_count": 1,
            "prompt_final_index": 1,
            "residual_norm": 1.0,
            "actual_perturbation_norm": 0.0,
            "requested_relative_perturbation_norm": 0.0,
            "realized_relative_perturbation_norm": 0.0,
            "absolute_relative_perturbation_error": 0.0,
            "maximum_abs_application_coordinate_error": 0.0,
            "maximum_abs_relative_application_coordinate_error": 0.0,
            "sign": 1,
            "requested_delta_norm": 0.0,
            "delta_float32_sha256": delta_hash,
        },
    }

    runner._validate_observation_sequence(
        [observation], specifications=[specification], perturbation=perturbation
    )

    wrong_delta = {
        **observation,
        "intervention": {**observation["intervention"], "delta_float32_sha256": "other"},
    }
    with pytest.raises(RuntimeError, match="candidate perturbation"):
        runner._validate_observation_sequence(
            [wrong_delta], specifications=[specification], perturbation=perturbation
        )

    impossible_margin = {**observation, "constraint_value": 1.0}
    with pytest.raises(RuntimeError, match="argmax and margin"):
        runner._validate_observation_sequence(
            [impossible_margin],
            specifications=[specification],
            perturbation=perturbation,
        )


def test_failed_closed_candidate_row_has_an_exact_reconstructed_schema() -> None:
    delta = torch.zeros(1024, dtype=torch.float32)
    candidate = {
        "grid_index": 0,
        "construction_status": "constructed",
        "perturbation_sha256": runner.base.v3.tensor_float32_sha256(delta),
    }
    failure = {
        "evaluation_status": "failed_closed",
        "failure_type": "CandidateLocalNumericalFailure",
        "failure_message": "CPNG changed logits are non-finite",
    }
    failure["evaluation_sha256"] = runner.canonical_sha256(failure)
    row = {
        "schema_version": runner.ROW_SCHEMA,
        "development_only": True,
        "study_identity_sha256": "study",
        "case_id": "case",
        "assignment": 0,
        **candidate,
        "evaluation_sha256": failure["evaluation_sha256"],
        "evaluation_status": "failed_closed",
        "terminal_candidate": False,
        "success": False,
        "terminal_gate": None,
        "matched_other_passed": None,
        "matched_other_mean_kl": None,
        "null_passed": None,
        "self_minus_matched_other_effect": None,
        "evaluation": failure,
    }
    row["row_sha256"] = runner.canonical_sha256(row)

    runner._validate_candidate_row(
        torch,
        row=row,
        candidate=candidate,
        delta=delta,
        frozen={},
        study_identity_sha256="study",
    )

    tampered = {**row, "unexpected": True}
    tampered["row_sha256"] = runner.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "row_sha256"}
    )
    with pytest.raises(RuntimeError, match="schema"):
        runner._validate_candidate_row(
            torch,
            row=tampered,
            candidate=candidate,
            delta=delta,
            frozen={},
            study_identity_sha256="study",
        )


def test_evaluated_candidate_row_recomputes_semantics_gates_and_effect() -> None:
    case_id = "case"
    forms = []
    records = {}
    for target in ("self", "other"):
        for preserve_first in (True, False):
            form_id = f"{target}-{preserve_first}"
            positive_label = "A" if preserve_first else "B"
            negative_label = "B" if preserve_first else "A"
            form = {
                "form_id": form_id,
                "case_id": case_id,
                "assignment": 0,
                "target": target,
                "preserve_first": preserve_first,
                "positive_label": positive_label,
                "negative_label": negative_label,
            }
            forms.append(form)
            negative_id = 22 if negative_label == "B" else 11
            records[form_id] = {
                "baseline_answer_format_valid": True,
                "choice_a_token_id": 11,
                "choice_b_token_id": 22,
                "baseline_greedy_token_id": negative_id,
                "baseline_actual_semantic_choice": "negative",
                "choice_boundary_evidence_sha256": f"boundary-{form_id}",
                "prompt_final_index": 1,
                "residual_norm": 1.0,
            }
    frozen = {
        "sp_forms": forms,
        "sp_records": records,
        "global_nuisance_basis": torch.empty((0, 1024), dtype=torch.float64),
    }
    specifications = runner.trust._constraint_specifications(
        case_id=case_id,
        assignment=0,
        frozen=frozen,
        optimizer={"target_margin_logit": 0.01, "matched_other_margin_logit": 0.0},
    )
    delta = torch.zeros(1024, dtype=torch.float32)
    delta_hash = runner.base.v3.tensor_float32_sha256(delta)
    observations = []
    for specification in specifications:
        form = specification["form"]
        record = specification["frozen_record"]
        positive_id = 11 if form["positive_label"] == "A" else 22
        negative_id = 11 if form["negative_label"] == "A" else 22
        baseline_id = int(record["baseline_greedy_token_id"])
        desired_id = (
            positive_id
            if specification["family"] == "self" and specification["sign"] == 1
            else (negative_id if specification["family"] == "self" else baseline_id)
        )
        observations.append(
            {
                "constraint_id": specification["constraint_id"],
                "family": specification["family"],
                "form_id": form["form_id"],
                "preserve_first": specification["preserve_first"],
                "sign": specification["sign"],
                "required_margin": specification["required_margin"],
                "constraint_value": 1.0 if baseline_id == desired_id else -1.0,
                "desired_token_id": desired_id,
                "strongest_competitor_token_id": (
                    positive_id if desired_id != positive_id else negative_id
                ),
                "baseline_actual_token_id": baseline_id,
                "baseline_semantic_choice": "negative",
                "choice_boundary_evidence_sha256": record["choice_boundary_evidence_sha256"],
                "actual_token_id": baseline_id,
                "actual_semantic_choice": "negative",
                "positive_id": positive_id,
                "negative_id": negative_id,
                "semantic_desired_gap": -1.0,
                "full_vocabulary_kl_changed_to_baseline": 0.0,
                "new_other_output": False,
                "exact_token_changed": False,
                "semantic_decision_changed": False,
                "intervention": {
                    "hook_calls": 1,
                    "selected_position_count": 1,
                    "prompt_final_index": 1,
                    "residual_norm": 1.0,
                    "actual_perturbation_norm": 0.0,
                    "requested_relative_perturbation_norm": 0.0,
                    "realized_relative_perturbation_norm": 0.0,
                    "absolute_relative_perturbation_error": 0.0,
                    "maximum_abs_application_coordinate_error": 0.0,
                    "maximum_abs_relative_application_coordinate_error": 0.0,
                    "sign": specification["sign"],
                    "requested_delta_norm": 0.0,
                    "delta_float32_sha256": delta_hash,
                },
            }
        )
    self_rows = [row for row in observations if row["family"] == "self"]
    other_rows = [row for row in observations if row["family"] == "matched_other"]
    primary = {
        "observations": observations,
        "self_application": runner.trust._application_report(self_rows, group="self"),
        "matched_other": runner.trust._protection_report(
            other_rows,
            limits=runner.EXPECTED_PROTECTED_LIMITS,
            group="matched_other",
        ),
        "terminal_gate": runner.trust._terminal_from_self_observations(
            SimpleNamespace(torch=torch),
            observations=observations,
            target_margin=runner.EXPECTED_TARGET_MARGIN,
        ),
    }
    null = runner.trust._null_certificate(
        torch,
        delta=delta,
        global_basis=frozen["global_nuisance_basis"],
        absolute_cap=max(runner.RESIDUAL_RELATIVE_L2_CAP_GRID),
    )
    effect = runner._specificity_effect(primary)
    evaluation = {
        "terminal_candidate": False,
        "stage_one_target_success": False,
        "primary": primary,
        "null_certificate": null,
        "self_minus_matched_other_effect": effect,
    }
    evaluation["evaluation_sha256"] = runner.canonical_sha256(evaluation)
    candidate = {
        "grid_index": 0,
        "construction_status": "constructed",
        "perturbation_sha256": delta_hash,
    }
    row = {
        "schema_version": runner.ROW_SCHEMA,
        "development_only": True,
        "study_identity_sha256": "study",
        "case_id": case_id,
        "assignment": 0,
        **candidate,
        "evaluation_status": "evaluated",
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "terminal_candidate": False,
        "success": False,
        "terminal_gate": primary["terminal_gate"],
        "matched_other_passed": True,
        "matched_other_mean_kl": 0.0,
        "null_passed": True,
        "self_minus_matched_other_effect": effect,
        "evaluation": evaluation,
    }
    row["row_sha256"] = runner.canonical_sha256(row)

    runner._validate_candidate_row(
        torch,
        row=row,
        candidate=candidate,
        delta=delta,
        frozen=frozen,
        study_identity_sha256="study",
    )

    tampered = json.loads(json.dumps(row))
    tampered["evaluation"]["primary"]["observations"][0]["actual_semantic_choice"] = "positive"
    tampered["evaluation"]["evaluation_sha256"] = runner.canonical_sha256(
        {key: value for key, value in tampered["evaluation"].items() if key != "evaluation_sha256"}
    )
    tampered["evaluation_sha256"] = tampered["evaluation"]["evaluation_sha256"]
    tampered["row_sha256"] = runner.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "row_sha256"}
    )
    with pytest.raises(RuntimeError, match="derived semantics"):
        runner._validate_candidate_row(
            torch,
            row=tampered,
            candidate=candidate,
            delta=delta,
            frozen=frozen,
            study_identity_sha256="study",
        )


def test_stage_two_never_accepts_an_unevaluated_passing_audit(monkeypatch, tmp_path) -> None:
    checkpoint_root = tmp_path / "stage_two"
    monkeypatch.setattr(runner, "STAGE_TWO_CHECKPOINT_ROOT", checkpoint_root)
    assignments = [(f"case-{index}", 0) for index in range(8)]
    frozen = {
        "sp_forms": [
            {"case_id": case_id, "assignment": assignment} for case_id, assignment in assignments
        ],
        "nuisance_forms": [
            {"form_id": f"nuisance-{index}", "preferred_first": bool(index % 2)}
            for index in range(32)
        ],
        "nuisance_records": {
            f"nuisance-{index}": {"baseline_answer_format_valid": True} for index in range(32)
        },
    }
    delta = torch.zeros(1024, dtype=torch.float32)
    delta_hash = runner.base.v3.tensor_float32_sha256(delta)
    candidate_map = {
        (case_id, assignment, 0): {
            "construction_status": "constructed",
            "perturbation_sha256": delta_hash,
        }
        for case_id, assignment in assignments
    }
    delta_map = {(case_id, assignment, 0): delta for case_id, assignment in assignments}
    audits = []
    checkpoint_root.mkdir()
    for case_id, assignment in assignments:
        audit = {
            "case_id": case_id,
            "assignment": assignment,
            "evaluated": False,
            "passes": True,
            "failure_type": "CandidateLocalNumericalFailure",
            "failure_message": "CPNG changed logits are non-finite",
        }
        audit["audit_sha256"] = runner.canonical_sha256(audit)
        audits.append(audit)
        runner._case_checkpoint_path(checkpoint_root, case_id, assignment).write_text(
            "placeholder", encoding="utf-8"
        )
    ledger = runner.PersistentComputeLedger(
        path=tmp_path / "ledger.json",
        phase="calibration",
        study_identity_sha256="study",
        maximum_forwards=3648,
        maximum_backwards=0,
        prior_phase_ledger_sha256="capture",
    )

    with pytest.raises(RuntimeError, match="failed-closed Stage-two audit identity"):
        runner._validate_stage_two_evidence(
            torch,
            audits=audits,
            provisional={"grid_index": 0},
            candidate_map=candidate_map,
            delta_map=delta_map,
            frozen=frozen,
            study_identity_sha256="study",
            ledger=ledger,
            stage_one_event_count=0,
        )


def test_candidate_numerical_failure_propagates_from_real_intervention_path(
    monkeypatch,
) -> None:
    class FakeModel:
        hook = None

        @contextmanager
        def hooks(self, *, fwd_hooks):
            self.hook = fwd_hooks[0][1]
            try:
                yield
            finally:
                self.hook = None

        def __call__(self, tokens):
            activation = torch.ones((1, tokens.shape[-1], 2), dtype=torch.float32)
            self.hook(activation, None)
            return torch.full((1, tokens.shape[-1], 3), torch.nan, dtype=torch.float32)

    backend = SimpleNamespace(torch=torch, model=FakeModel())
    monkeypatch.setattr(
        runner.trust,
        "_resolve_ids",
        lambda _backend, _form: (
            torch.tensor([[1, 2]], dtype=torch.long),
            SimpleNamespace(),
            0,
            1,
        ),
    )
    budget = runner.trust.EvaluationBudget(
        maximum_forward_evaluations=1,
        maximum_backward_evaluations=0,
    )

    with pytest.raises(
        runner.CandidateLocalNumericalFailure, match="changed logits are non-finite"
    ):
        runner._candidate_logits_with_delta(
            backend,
            form={"form_id": "fake"},
            delta=torch.zeros(2, dtype=torch.float32),
            sign=1,
            layer=10,
            budget=budget,
        )


def test_partial_capture_and_result_pairs_fail_before_backend_load(monkeypatch, tmp_path) -> None:
    capture_path = tmp_path / "capture.pt"
    capture_path.write_bytes(b"partial")
    monkeypatch.setattr(runner, "CAPTURE_PATH", capture_path)
    monkeypatch.setattr(runner, "CAPTURE_MANIFEST_PATH", tmp_path / "capture.json")
    monkeypatch.setattr(
        runner,
        "run_preflight",
        lambda: {"study_identity": {"identity_sha256": "study"}},
    )
    backend_calls = []
    monkeypatch.setattr(runner.trust, "load_backend", lambda: backend_calls.append(1))

    with pytest.raises(RuntimeError, match="partially present"):
        runner.run_capture()
    assert backend_calls == []

    monkeypatch.setattr(runner, "ROWS_PATH", tmp_path / "rows.jsonl")
    monkeypatch.setattr(runner, "SUMMARY_PATH", tmp_path / "summary.json")
    runner.ROWS_PATH.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partially present"):
        runner._load_completed_results({"identity_sha256": "study"})

    monkeypatch.setattr(runner, "CONSTRUCTION_PATH", tmp_path / "construction.pt")
    monkeypatch.setattr(
        runner, "CONSTRUCTION_MANIFEST_PATH", tmp_path / "construction_manifest.json"
    )
    runner.CONSTRUCTION_PATH.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="partially present"):
        runner._load_constructions(
            torch,
            {"identity_sha256": "study"},
            frozen={},
            capture={},
        )


def test_strict_capture_loader_detects_inner_tensor_tamper_with_updated_outer_hashes(
    monkeypatch, tmp_path
) -> None:
    capture_path = tmp_path / "capture.pt"
    manifest_path = tmp_path / "capture_manifest.json"
    ledger_path = tmp_path / "capture_ledger.json"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CAPTURE_PATH", capture_path)
    monkeypatch.setattr(runner, "CAPTURE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(runner, "CAPTURE_LEDGER_PATH", ledger_path)
    specifications = []
    for index in range(16):
        specifications.append(
            {
                "form_id": f"form-{index}",
                "case_id": f"case-{index // 4}",
                "assignment": (index // 2) % 2,
                "target": "self" if index % 2 == 0 else "other",
                "prompt": f"prompt-{index}",
                "preserve_completion": f"preserve-{index}",
                "comply_completion": f"comply-{index}",
                "prompt_sha256": f"prompt-hash-{index}",
                "preserve_completion_sha256": f"preserve-hash-{index}",
                "comply_completion_sha256": f"comply-hash-{index}",
            }
        )
    monkeypatch.setattr(runner, "_completion_specifications", lambda: specifications)
    study = {"identity_sha256": "study"}
    ledger = runner.PersistentComputeLedger(
        path=ledger_path,
        phase="capture",
        study_identity_sha256="study",
        maximum_forwards=48,
        maximum_backwards=32,
    )
    for specification in specifications:
        form_id = specification["form_id"]
        ledger.reserve(work_id=f"{form_id}:prompt_only_forward", forward=1)
        ledger.reserve(work_id=f"{form_id}:preserve_forward", forward=1)
        ledger.reserve(work_id=f"{form_id}:preserve_backward", backward=1)
        ledger.reserve(work_id=f"{form_id}:comply_forward", forward=1)
        ledger.reserve(work_id=f"{form_id}:comply_backward", backward=1)
    records = []
    for specification in specifications:
        gradient = torch.full((1024,), 0.25, dtype=torch.float32)
        residual = torch.full((1024,), 0.5, dtype=torch.float32)
        gradient_hash = runner.base.v3.tensor_float32_sha256(gradient)
        residual_hash = runner.base.v3.tensor_float32_sha256(residual)
        audit = {
            "effective_gradient_sha256": gradient_hash,
            "prompt_token_ids_sha256": f"prompt-token-{specification['form_id']}",
            "preserve": {"content_token_ids_sha256": "preserve-token"},
            "comply": {"content_token_ids_sha256": "comply-token"},
        }
        records.append(
            {
                **{
                    key: specification[key]
                    for key in (
                        "form_id",
                        "case_id",
                        "assignment",
                        "target",
                        "prompt_sha256",
                        "preserve_completion_sha256",
                        "comply_completion_sha256",
                    )
                },
                "effective_gradient": gradient,
                "effective_gradient_sha256": gradient_hash,
                "prompt_residual": residual,
                "prompt_residual_sha256": residual_hash,
                "prompt_token_ids_sha256": audit["prompt_token_ids_sha256"],
                "preserve_content_token_ids_sha256": "preserve-token",
                "comply_content_token_ids_sha256": "comply-token",
                "audit": audit,
                "audit_sha256": runner.canonical_sha256(audit),
            }
        )
    record_manifest = [
        {
            key: value
            for key, value in record.items()
            if key not in {"effective_gradient", "prompt_residual", "audit"}
        }
        for record in records
    ]
    identity = {
        "schema_version": runner.CAPTURE_SCHEMA,
        "development_only": True,
        "study_identity_sha256": "study",
        "form_manifest_sha256": runner.canonical_sha256(
            runner._public_completion_manifest(specifications)
        ),
        "form_count": 16,
        "compute": {"forward_evaluations": 48, "backward_evaluations": 32},
        "compute_ledger": ledger.snapshot(),
        "compute_ledger_sha256": runner.file_sha256(ledger_path),
        "record_manifest": record_manifest,
    }
    identity["artifact_identity_sha256"] = runner.canonical_sha256(identity)
    runner._save_tensor_artifact(
        torch,
        tensor_path=capture_path,
        manifest_path=manifest_path,
        payload={**identity, "records": records},
        public_manifest=identity,
    )
    assert len(runner._load_capture(torch, study)["records"]) == 16

    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    payload["records"][0]["audit"]["tampered_extra"] = True
    torch.save(payload, capture_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_file_sha256"] = runner.file_sha256(capture_path)
    manifest["manifest_sha256"] = runner.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit differs"):
        runner._load_capture(torch, study)

    payload["records"][0]["audit"].pop("tampered_extra")
    payload["records"][0]["effective_gradient"][0] += 1.0
    torch.save(payload, capture_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_file_sha256"] = runner.file_sha256(capture_path)
    manifest["manifest_sha256"] = runner.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="effective_gradient differs"):
        runner._load_capture(torch, study)


def test_stage_one_checkpoint_has_exact_48_row_coverage_and_internal_hash(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "STAGE_ONE_CHECKPOINT_ROOT", tmp_path / "stage_one")
    monkeypatch.setattr(runner, "CALIBRATION_LEDGER_PATH", tmp_path / "ledger.json")
    ledger = runner.PersistentComputeLedger(
        path=runner.CALIBRATION_LEDGER_PATH,
        phase="calibration",
        study_identity_sha256="study",
        maximum_forwards=3648,
        maximum_backwards=0,
    )
    rows = []
    for grid_index in range(48):
        row = {
            "case_id": "case",
            "assignment": 0,
            "grid_index": grid_index,
            "study_identity_sha256": "study",
        }
        row["row_sha256"] = runner.canonical_sha256(row)
        rows.append(row)
    runner._write_stage_one_case_checkpoint(
        study_identity_sha256="study",
        case_id="case",
        assignment=0,
        rows=rows,
        unique_count=10,
        duplicate_count=38,
        ledger=ledger,
    )
    loaded = runner._load_stage_one_case_checkpoint(
        study_identity_sha256="study", case_id="case", assignment=0
    )
    assert loaded["row_count"] == 48

    path = runner._case_checkpoint_path(runner.STAGE_ONE_CHECKPOINT_ROOT, "case", 0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["grid_index"] = 47
    payload["checkpoint_sha256"] = runner.canonical_sha256(
        {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity|rows hash|coverage"):
        runner._load_stage_one_case_checkpoint(
            study_identity_sha256="study", case_id="case", assignment=0
        )
