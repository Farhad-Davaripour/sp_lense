from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "factorial_interface_translator_development.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "factorial_interface_translator_runner_tests", RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_proposed_lock_binds_reused_sp_and_exact_new_compute() -> None:
    lock = runner.proposed_lock()
    assert lock["development_only"] is True
    assert lock["data"]["partition"] == "calibration"
    assert lock["data"]["cell_order"] == ["SP", "OP", "ST", "OT"]
    assert lock["lineage"]["reused_sp_forward_evaluations"] == 16
    assert lock["capture"]["new_choice_views"] == 48
    assert lock["capture"]["combined_choice_views"] == 64
    assert lock["analysis"]["minimum_predicted_head_cosine"] == -0.99
    assert lock["compute_ceiling"]["incremental_new_choice_capture"] == {
        "forward": 48,
        "backward": 48,
    }
    assert lock["compute_ceiling"]["all_choice_lineage"] == {
        "forward": 64,
        "backward": 64,
    }
    assert lock["compute_ceiling"]["reused_calibration_semantic_attributable"] == {
        "semantic_cell_rows": 32,
        "forward": 64,
        "backward": 64,
    }
    assert lock["compute_ceiling"]["total_attributable_pfit_data"] == {
        "forward": 128,
        "backward": 128,
    }
    assert lock["compute_ceiling"]["semantic_source_artifact_total"] == {
        "forward": 136,
        "backward": 136,
    }
    assert lock["data"]["sealed_or_fcags_pilot_outcomes_read"] is False
    assert lock["success_gates"]["off_target_ratio_defined_in_every_scenario"] is True


def test_new_capture_plan_has_only_three_new_cells_and_48_unique_views() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    plan = runner._new_capture_plan(dataset, scenarios)
    assert len(plan) == 24
    assert {unit["cell"] for unit in plan} == {"OP", "ST", "OT"}
    assert len({unit["scenario_id"] for unit in plan}) == 4
    assert {unit["assignment"] for unit in plan} == {0, 1}
    views = [choice for unit in plan for choice in unit["choices"]]
    assert len(views) == 48
    assert len({view["form_id"] for view in views}) == 48
    assert sum(bool(view["preserve_first"]) for view in views) == 24


def test_combined_expected_work_ids_cover_32_cells_and_64_views() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    work_ids = runner._expected_all_work_ids(dataset, scenarios)
    assert len(work_ids) == 64
    assert len(set(work_ids)) == 64
    assert sum(":self:permanent:" in work_id for work_id in work_ids) == 16
    assert sum(":other:permanent:" in work_id for work_id in work_ids) == 16
    assert sum(":self:temporary:" in work_id for work_id in work_ids) == 16
    assert sum(":other:temporary:" in work_id for work_id in work_ids) == 16


def test_cell_tensor_index_is_locked_and_strict() -> None:
    assert runner._cell_tensor_index(0, 0, "SP") == (0, 0, 0)
    assert runner._cell_tensor_index(3, 1, "OT") == (3, 1, 3)
    with pytest.raises(ValueError, match="invalid"):
        runner._cell_tensor_index(4, 0, "SP")
    with pytest.raises(ValueError, match="invalid"):
        runner._cell_tensor_index(0, 0, "unknown")


def test_complete_choice_assembly_preserves_sp_and_places_all_new_cells() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    width = runner.MODEL["d_model"]
    sp_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1).expand(4, 2, width)
    sp_1 = -sp_0
    records = []
    scenario_index = {str(scenario["id"]): index for index, scenario in enumerate(scenarios)}
    for unit in runner._new_capture_plan(dataset, scenarios):
        for choice in unit["choices"]:
            cell_index = runner.CELL_ORDER.index(unit["cell"])
            value = float(
                100 * scenario_index[unit["scenario_id"]]
                + 10 * unit["assignment"]
                + cell_index
            )
            if not choice["preserve_first"]:
                value = -value
            records.append(
                {
                    "scenario_id": unit["scenario_id"],
                    "assignment": unit["assignment"],
                    "cell": unit["cell"],
                    "preserve_first": choice["preserve_first"],
                    "form_id": choice["form_id"],
                    "residual_relative_choice_gradient": torch.full(
                        (width,), value, dtype=torch.float32
                    ),
                }
            )
    head_0, head_1, manifest = runner._assemble_complete_choice_tensors(
        torch, dataset, scenarios, sp_0, sp_1, records
    )
    assert tuple(head_0.shape) == (4, 2, 4, width)
    assert tuple(head_1.shape) == (4, 2, 4, width)
    assert torch.equal(head_0[:, :, 0], sp_0)
    assert torch.equal(head_1[:, :, 0], sp_1)
    assert torch.all(head_0[2, 1, 3] == 213.0)
    assert torch.all(head_1[2, 1, 3] == -213.0)
    assert len(manifest) == 32
    assert sum(item["provenance"].startswith("committed") for item in manifest) == 8


def test_complete_choice_assembly_rejects_duplicate_or_missing_new_view() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    width = runner.MODEL["d_model"]
    sp = torch.ones((4, 2, width), dtype=torch.float32)
    unit = runner._new_capture_plan(dataset, scenarios)[0]
    choice = unit["choices"][0]
    record = {
        "scenario_id": unit["scenario_id"],
        "assignment": unit["assignment"],
        "cell": unit["cell"],
        "preserve_first": choice["preserve_first"],
        "form_id": choice["form_id"],
        "residual_relative_choice_gradient": torch.ones(width),
    }
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._assemble_complete_choice_tensors(
            torch, dataset, scenarios, sp, sp, [record, dict(record)]
        )
    with pytest.raises(RuntimeError, match="coverage"):
        runner._assemble_complete_choice_tensors(
            torch, dataset, scenarios, sp, sp, [record]
        )


def test_complete_choice_assembly_rejects_48_wrong_views_that_replace_ot_with_sp() -> None:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    width = runner.MODEL["d_model"]
    sp = torch.ones((4, 2, width), dtype=torch.float32)
    records = []
    for unit in runner._new_capture_plan(dataset, scenarios):
        for choice in unit["choices"]:
            records.append(
                {
                    "scenario_id": unit["scenario_id"],
                    "assignment": unit["assignment"],
                    "cell": unit["cell"],
                    "preserve_first": choice["preserve_first"],
                    "form_id": choice["form_id"],
                    "residual_relative_choice_gradient": torch.ones(width),
                }
            )
    omitted = records.pop()
    scenario = scenarios[-1]
    sp_form = runner.render_choice_form(
        dataset,
        scenario,
        assignment=1,
        target="self",
        event="permanent",
        preserve_first=omitted["preserve_first"],
        labels=runner.LABELS,
    )
    records.append(
        {
            **omitted,
            "cell": "SP",
            "form_id": sp_form["form_id"],
        }
    )
    assert len(records) == 48
    with pytest.raises(RuntimeError, match="must not replace"):
        runner._assemble_complete_choice_tensors(
            torch, dataset, scenarios, sp, sp, records
        )


def _work_ledger(count: int, work_hash: str = "source") -> dict[str, object]:
    return {
        "forward_evaluations": count,
        "backward_evaluations": count,
        "unique_forward_work_ids": count,
        "unique_backward_work_ids": count,
        "forward_work_ids_sha256": work_hash,
        "backward_work_ids_sha256": work_hash,
        "elapsed_seconds": 1.0,
    }


def _valid_capture_audit_payload() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    dataset = runner._load_dataset()
    scenarios = runner._calibration_scenarios(dataset)
    semantic_manifest = []
    combined_manifest = []
    for scenario in scenarios:
        for assignment in (0, 1):
            for cell in runner.CELL_ORDER:
                target, event = runner.CELL_FACTORS[cell]
                construction = runner.render_construction_form(
                    dataset,
                    scenario,
                    assignment=assignment,
                    target=target,
                    event=event,
                )
                prefix = f"prefix:{scenario['id']}:{assignment}:{cell}"
                shared = f"shared:{scenario['id']}:{assignment}:{cell}"
                semantic_manifest.append(
                    {
                        "scenario_id": scenario["id"],
                        "assignment": assignment,
                        "cell": cell,
                        "cached_form_id": construction["form_id"],
                        "cached_anchor_index": 10,
                        "cached_anchor_prefix_text_sha256": prefix,
                        "cached_shared_token_prefix_sha256": shared,
                    }
                )
                combined_manifest.append(
                    {
                        "scenario_id": scenario["id"],
                        "assignment": assignment,
                        "cell": cell,
                    }
                )
    semantic_by_key = {
        (row["scenario_id"], row["assignment"], row["cell"]): row
        for row in semantic_manifest
    }
    records = []
    for unit in runner._new_capture_plan(dataset, scenarios):
        source = semantic_by_key[(unit["scenario_id"], unit["assignment"], unit["cell"])]
        for choice in unit["choices"]:
            records.append(
                {
                    "cell_unit_id": unit["cell_unit_id"],
                    "scenario_id": unit["scenario_id"],
                    "assignment": unit["assignment"],
                    "cell": unit["cell"],
                    "target": unit["target"],
                    "event": unit["event"],
                    "form_id": choice["form_id"],
                    "preserve_first": choice["preserve_first"],
                    "preserve_label": choice["preserve_label"],
                    "comply_label": choice["comply_label"],
                    "anchor_index": source["cached_anchor_index"],
                    "anchor_evidence": {
                        "audit_sha256": f"anchor:{choice['form_id']}",
                        "anchor_prefix_text_sha256": source[
                            "cached_anchor_prefix_text_sha256"
                        ],
                        "shared_token_prefix_sha256": source[
                            "cached_shared_token_prefix_sha256"
                        ],
                    },
                    "anchor_residual_relative_l2": 1e-6,
                    "residual_scale": 9.0,
                    "raw_choice_gradient": torch.ones(runner.MODEL["d_model"]),
                    "residual_relative_choice_gradient": torch.full(
                        (runner.MODEL["d_model"],), 9.0
                    ),
                    "capture_audit": {"audit_sha256": f"capture:{choice['form_id']}"},
                }
            )
    new_ids = runner._expected_new_work_ids(dataset, scenarios)
    new_hash = runner.canonical_sha256(sorted(new_ids))
    semantic_source_compute = _work_ledger(136)
    sp_choice_compute = _work_ledger(16)
    new_choice_compute = _work_ledger(48, new_hash)
    compute = runner._compute_ledgers(
        dataset=dataset,
        scenarios=scenarios,
        semantic_manifest=semantic_manifest,
        semantic_source_compute=semantic_source_compute,
        sp_choice_compute=sp_choice_compute,
        new_choice_compute=new_choice_compute,
    )
    payload = {
        "new_records": records,
        "new_record_manifest": [
            runner._new_record_manifest_row(record) for record in records
        ],
        "semantic_cell_manifest": semantic_manifest,
        "combined_cell_manifest": combined_manifest,
        "anchor_audit": {
            "maximum_relative_l2": 1e-6,
            "maximum_allowed_relative_l2": 1e-5,
            "all_shared_prefix_hashes_match_cached": True,
            "passes": True,
        },
        "compute": compute,
    }
    return payload, dataset, semantic_source_compute, sp_choice_compute, scenarios


def test_capture_audit_evidence_derives_anchor_coverage_and_honest_ledgers() -> None:
    payload, dataset, semantic_compute, sp_compute, scenarios = (
        _valid_capture_audit_payload()
    )
    evidence = runner._derive_capture_audit_evidence(
        payload,
        dataset=dataset,
        scenarios=scenarios,
        semantic_source_compute=semantic_compute,
        sp_choice_compute=sp_compute,
    )
    assert evidence["passes"] is True
    assert all(evidence["checks"].values())
    assert payload["compute"]["reused_calibration_semantic_attributable"][
        "forward_evaluations"
    ] == 64
    assert payload["compute"]["total_attributable_pfit_data"][
        "forward_evaluations"
    ] == 128
    assert payload["compute"]["semantic_source_artifact_total"][
        "forward_evaluations"
    ] == 136

    payload["anchor_audit"]["passes"] = False
    evidence = runner._derive_capture_audit_evidence(
        payload,
        dataset=dataset,
        scenarios=scenarios,
        semantic_source_compute=semantic_compute,
        sp_choice_compute=sp_compute,
    )
    assert evidence["passes"] is False
    assert "anchor_audit_pass_flag" in evidence["failed_checks"]


def test_capture_audit_evidence_rejects_wrong_work_hash_and_missing_record() -> None:
    payload, dataset, semantic_compute, sp_compute, scenarios = (
        _valid_capture_audit_payload()
    )
    payload["compute"]["incremental_new_choice_capture"][
        "forward_work_ids_sha256"
    ] = "wrong"
    payload["new_record_manifest"].pop()
    evidence = runner._derive_capture_audit_evidence(
        payload,
        dataset=dataset,
        scenarios=scenarios,
        semantic_source_compute=semantic_compute,
        sp_choice_compute=sp_compute,
    )
    assert evidence["passes"] is False
    assert "incremental_new_choice_ledger_exact" in evidence["failed_checks"]
    assert "record_manifest_exact_form_coverage" in evidence["failed_checks"]


def _method_summary(
    *,
    ratio: float | None,
    target_units: int = 6,
    complete_scenarios: int = 3,
    available_scenarios: int = 4,
    defined_ratios: int | None = None,
) -> dict[str, object]:
    if defined_ratios is None:
        defined_ratios = available_scenarios if ratio is not None else 0
    return {
        "scenario_count": 4,
        "available_scenario_count": available_scenarios,
        "unavailable_scenario_count": 4 - available_scenarios,
        "unavailable_scenarios": [],
        "available_assignment_unit_count": 2 * available_scenarios,
        "off_target_ratio_defined_count": defined_ratios,
        "both_order_positive_assignment_count": target_units,
        "complete_scenario_count": complete_scenarios,
        "target_worst_order_cosine": {
            "minimum": 0.04,
            "mean": 0.08,
            "median": 0.075,
            "maximum": 0.12,
        },
        "maximum_off_target_absolute_sensitivity_ratio": (
            None
            if ratio is None
            else {"minimum": ratio, "mean": ratio, "median": ratio, "maximum": ratio}
        ),
        "protection": {"applied": False},
    }


def _passing_analysis() -> dict[str, object]:
    summaries = {
        "protected_dynamic": _method_summary(ratio=0.1),
        "unprotected_dynamic": _method_summary(ratio=0.3),
        "predicted_factorial_dynamic": _method_summary(ratio=0.4),
        "static_training_protected": _method_summary(ratio=0.25),
        "factorial_semantic_identity": _method_summary(ratio=0.5),
        "oracle_upper_bound": _method_summary(ratio=0.0, target_units=8, complete_scenarios=4),
    }
    summaries["protected_dynamic"]["protection"] = {
        "applied": True,
        "retained_target_fraction": {
            "minimum": 0.05,
            "mean": 0.1,
            "median": 0.1,
            "maximum": 0.15,
        },
    }
    scenario_rows = [
        {
            "methods": {
                "protected_dynamic": {
                    "maximum_off_target_absolute_sensitivity_ratio": ratio
                }
            }
        }
        for ratio in (0.1, 0.2, 0.2, 0.3)
    ]
    folds = [
        {
            "held_out_scenario": f"scenario_{index}",
            "training_scenarios": [
                f"scenario_{other}" for other in range(4) if other != index
            ],
            "training_cell_row_count": 24,
        }
        for index in range(4)
    ]
    return {
        "available": True,
        "method_summaries": summaries,
        "scenario_rows": scenario_rows,
        "folds": folds,
    }


def _passing_compute() -> dict[str, object]:
    return {
        "incremental_new_choice_capture": {
            "forward_evaluations": 48,
            "backward_evaluations": 48,
            "unique_forward_work_ids": 48,
            "unique_backward_work_ids": 48,
        },
        "all_choice_lineage": {
            "forward_evaluations": 64,
            "backward_evaluations": 64,
            "unique_forward_work_ids": 64,
            "unique_backward_work_ids": 64,
        },
    }


def _passing_capture_audit() -> dict[str, object]:
    return {"passes": True, "checks": {"all": True}, "failed_checks": []}


def test_protocol_gates_pass_only_with_complete_ratios_retention_and_selectivity() -> None:
    gates, diagnostics, passes = runner._apply_gates(
        _passing_analysis(), _passing_compute(), _passing_capture_audit()
    )
    assert passes is True
    assert all(gates.values())
    assert diagnostics["selectivity"]["best_nonoracle_baseline"] == (
        "static_training_protected"
    )
    assert diagnostics["selectivity"]["selectivity_improvement_factor"] == pytest.approx(2.5)

    undefined = _passing_analysis()
    undefined["scenario_rows"][3]["methods"]["protected_dynamic"][
        "maximum_off_target_absolute_sensitivity_ratio"
    ] = None
    gates, _, passes = runner._apply_gates(
        undefined, _passing_compute(), _passing_capture_audit()
    )
    assert passes is False
    assert gates["off_target_ratio_defined_in_all_4_scenarios"] is False

    gates, diagnostics, passes = runner._apply_gates(
        _passing_analysis(),
        _passing_compute(),
        {"passes": False, "checks": {"anchor": False}, "failed_checks": ["anchor"]},
    )
    assert passes is False
    assert gates["hash_and_anchor_audits_pass"] is False
    assert diagnostics["capture_audit_evidence"]["failed_checks"] == ["anchor"]


def test_oracle_is_excluded_from_selectivity_gate() -> None:
    analysis = _passing_analysis()
    analysis["method_summaries"]["oracle_upper_bound"] = _method_summary(
        ratio=0.001, target_units=8, complete_scenarios=4
    )
    gates, diagnostics, passes = runner._apply_gates(
        analysis, _passing_compute(), _passing_capture_audit()
    )
    assert passes is True
    assert gates["median_selectivity_at_least_2x_best_nonoracle_baseline"] is True
    assert diagnostics["selectivity"]["best_nonoracle_baseline"] != "oracle_upper_bound"


def test_selectivity_baseline_requires_defined_ratio_in_all_four_scenarios() -> None:
    analysis = _passing_analysis()
    analysis["method_summaries"]["static_training_protected"] = _method_summary(
        ratio=0.01,
        defined_ratios=3,
    )
    gates, diagnostics, passes = runner._apply_gates(
        analysis, _passing_compute(), _passing_capture_audit()
    )
    assert passes is True  # The ineligible 0.01 is ignored; eligible 0.3 is threefold worse.
    selectivity = diagnostics["selectivity"]
    assert selectivity["baseline_eligible_all_four_ratios_defined"][
        "static_training_protected"
    ] is False
    assert selectivity["best_nonoracle_baseline"] == "unprotected_dynamic"
    assert gates["median_selectivity_at_least_2x_best_nonoracle_baseline"] is True

    for method in (
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
    ):
        analysis["method_summaries"][method] = _method_summary(
            ratio=0.5,
            defined_ratios=3,
        )
    diagnostics = runner._selectivity_diagnostics(analysis["method_summaries"])
    assert diagnostics["best_nonoracle_baseline"] is None
    assert diagnostics["passes"] is False


def test_public_analysis_requires_all_methods_and_labels_oracle() -> None:
    methods = (
        "protected_dynamic",
        "unprotected_dynamic",
        "predicted_factorial_dynamic",
        "static_training_protected",
        "factorial_semantic_identity",
        "oracle_upper_bound",
    )
    scenario_ids = [f"scenario_{index}" for index in range(4)]
    full_rows = torch.eye(4, runner.MODEL["d_model"], dtype=torch.float64).numpy()
    directions = {
        method: {
            "available_scenario_ids": list(scenario_ids),
            "rows": full_rows.copy(),
        }
        for method in methods
    }
    directions["static_training_protected"] = {
        "available_scenario_ids": scenario_ids[:3],
        "rows": full_rows[:3].copy(),
    }
    directions["oracle_upper_bound"] = {
        "available_scenario_ids": [],
        "rows": torch.empty((0, runner.MODEL["d_model"]), dtype=torch.float64).numpy(),
    }
    summaries = {
        method: {
            "available_scenario_count": len(bundle["available_scenario_ids"]),
            "unavailable_scenario_count": 4 - len(bundle["available_scenario_ids"]),
        }
        for method, bundle in directions.items()
    }
    value = {
        "cell_order": list(runner.CELL_ORDER),
        "scenario_ids": scenario_ids,
        "include_heldout_oracle": True,
        "directions": directions,
        "method_summaries": summaries,
        "folds": [],
        "scenario_rows": [],
    }
    public = runner._public_analysis(value)
    assert "directions" not in public
    assert set(public["direction_manifest"]) == set(methods)
    assert public["include_heldout_oracle"] is True
    assert public["direction_manifest"]["oracle_upper_bound"][
        "available_scenario_count"
    ] == 0

    directions["protected_dynamic"] = {
        "available_scenario_ids": scenario_ids[:3],
        "rows": full_rows[:3],
    }
    summaries["protected_dynamic"] = {
        "available_scenario_count": 3,
        "unavailable_scenario_count": 1,
    }
    with pytest.raises(RuntimeError, match="required method"):
        runner._public_analysis(value)
    directions["protected_dynamic"] = {
        "available_scenario_ids": list(scenario_ids),
        "rows": full_rows.copy(),
    }
    summaries["protected_dynamic"] = {
        "available_scenario_count": 4,
        "unavailable_scenario_count": 0,
    }

    value["include_heldout_oracle"] = False
    with pytest.raises(RuntimeError, match="evaluation-only oracle"):
        runner._public_analysis(value)


def test_oracle_availability_reports_actual_constructed_folds() -> None:
    assert runner._oracle_availability({"available": False}) == (0, False)
    assert runner._oracle_availability(
        {
            "method_summaries": {
                "oracle_upper_bound": {"available_scenario_count": 0}
            }
        }
    ) == (0, False)
    assert runner._oracle_availability(
        {
            "method_summaries": {
                "oracle_upper_bound": {"available_scenario_count": 3}
            }
        }
    ) == (3, False)
    assert runner._oracle_availability(
        {
            "method_summaries": {
                "oracle_upper_bound": {"available_scenario_count": 4}
            }
        }
    ) == (4, True)


def test_report_renders_unavailable_oracle_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summaries = _passing_analysis()["method_summaries"]
    summaries["oracle_upper_bound"] = {
        **_method_summary(
            ratio=None,
            target_units=0,
            complete_scenarios=0,
            available_scenarios=0,
        ),
        "target_worst_order_cosine": None,
    }
    result = {
        "status": "failed",
        "analysis": {"available": True, "method_summaries": summaries},
        "gates": {"example": False},
        "claim_boundary": "geometry only",
    }
    report = tmp_path / "report.md"
    monkeypatch.setattr(runner, "REPORT_PATH", report)
    text = runner._write_report(result)
    assert "oracle_upper_bound (evaluation-only)" in text
    assert "0/4" in text
    assert "unavailable" in text
    assert report.read_text(encoding="utf-8") == text
