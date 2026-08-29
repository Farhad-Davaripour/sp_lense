from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "decision_margin_shield_layer_screen.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location("dms_layer_screen_tests", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _valid_chunk_material(specifications):
    gradients = torch.empty((len(specifications), 23, 1024), dtype=torch.float32)
    residuals = torch.empty_like(gradients)
    records = []
    for row_index, specification in enumerate(specifications):
        gradients[row_index].fill_(row_index + 0.25)
        residuals[row_index].fill_(row_index + 10.5)
        gradient_hash = runner.tensor_float32_sha256(gradients[row_index])
        residual_hash = runner.tensor_float32_sha256(residuals[row_index])
        public = runner._public_specification(specification)
        anchor_index = 7 + row_index
        prompt_token_hashes = [
            runner.canonical_sha256(["token-row", row_index, index])
            for index in range(len(specification["evidence_prompts"]))
        ]
        anchor_evidence = runner._with_hash(
            {
                "schema_version": "sp_lense.shared_causal_anchor_evidence.v1",
                "anchor_position": "last_token_of_longest_shared_prompt_prefix",
                "anchor_index": anchor_index,
                "shared_prefix_length": anchor_index + 1,
                "shared_token_prefix_sha256": runner.canonical_sha256(
                    ["shared", row_index]
                ),
                "prompt_token_sha256s": prompt_token_hashes,
                "prompt_count": len(prompt_token_hashes),
                "anchor_prefix_text_sha256": public["anchor_prefix_sha256"],
                "anchor_marker": "[CAUSAL DECISION ANCHOR]",
                "anchor_marker_present_in_decoded_shared_prefix": True,
                "future_suffix_cannot_change_anchor_by_causal_mask": True,
            },
            "audit_sha256",
        )
        baseline = 0.125 + row_index
        capture_audit = runner._with_hash(
            {
                "schema_version": "sp_lense.multilayer_choice_anchor_capture.v1",
                "objective": "preserve_label_minus_comply_label_next_token_logit",
                "preserve_label": public["preserve_label"],
                "comply_label": public["comply_label"],
                "preserve_log_odds": baseline,
                "anchor_index": anchor_index,
                "gradient_position": "shared_pre_encoding_causal_anchor",
                "layers": list(range(23)),
                "hook_call_counts": {str(layer): 1 for layer in range(23)},
                "raw_gradients_float32_sha256": gradient_hash,
                "anchor_residuals_float32_sha256": residual_hash,
                "choice_boundary_evidence_sha256": runner.canonical_sha256(
                    ["boundary", row_index]
                ),
                "prompt_token_ids_sha256": prompt_token_hashes[0],
                "model_parameter_gradients_allocated": False,
            },
            "audit_sha256",
        )
        records.append(
            {
                **public,
                "row_index": row_index,
                "anchor_index": anchor_index,
                "anchor_evidence": anchor_evidence,
                "capture_audit": capture_audit,
                "preserve_minus_comply_baseline_log_odds": baseline,
                "gradient_float32_sha256": gradient_hash,
                "anchor_residual_float32_sha256": residual_hash,
            }
        )
    return records, gradients, residuals


def _rehash_chunk_payload(payload):
    payload["tensor_hashes"] = {
        name: runner.tensor_float32_sha256(value)
        for name, value in sorted(payload["tensors"].items())
    }
    public = {key: value for key, value in payload.items() if key != "tensors"}
    public.pop("chunk_identity_sha256", None)
    payload["chunk_identity_sha256"] = runner.canonical_sha256(public)
    return payload


def _save_valid_test_chunk(tmp_path: Path, *, count: int = 2):
    specifications = runner._capture_specifications(runner._load_dataset())[:count]
    records, gradients, residuals = _valid_chunk_material(specifications)
    path = tmp_path / "chunk.pt"
    runner._save_tensor_chunk(
        torch,
        path=path,
        chunk_index=0,
        plan_sha256="plan-id",
        lock_identity_sha256="lock-id",
        records=records,
        gradients=gradients,
        residuals=residuals,
    )
    return path, specifications


def test_proposed_lock_binds_model_runtime_content_geometry_and_zero_outcomes() -> None:
    lock = runner.proposed_lock()
    assert lock["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert lock["runtime"]["torch"] == "2.13.0+cpu"
    assert lock["capture"]["layers"] == list(range(23))
    assert lock["capture"]["excluded_endpoint_layer"] == 23
    assert lock["capture"]["record_count"] == 136
    assert lock["capture"]["capture_plan_sha256"]
    assert lock["capture"]["prompt_content_sha256"]
    assert lock["dataset"]["selection_partition"] == "calibration_only"
    assert len(lock["dataset"]["calibration_scenario_ids"]) == 4
    assert len(lock["dataset"]["pilot_scenario_ids_captured_but_not_screened"]) == 4
    assert lock["geometry"]["cap_frontier"] == [1.0, 1.5, 2.0]
    assert lock["geometry"]["qualification_cap"] == 2.0
    assert "abs(b)=0.05_is_margin_certified" in lock["geometry"]["small_baseline_rule"]
    assert lock["geometry"]["finite_intervention_outcomes_inspected"] is False
    assert lock["geometry"]["pilot_geometry_computed"] is False
    assert lock["compute_ceiling"] == {
        "capture": {"forward": 136, "backward": 136},
        "geometry": {"forward": 0, "backward": 0},
        "finite_intervention": {"forward": 0, "backward": 0},
        "generated_tokens": 0,
    }
    assert {
        "dms_math",
        "cts_solver",
        "anchor_runtime",
        "factorial_math",
        "backend",
        "comparison_runtime",
        "comparison_intervention",
        "semantic_completion_gradient",
        "steering_methods",
        "gradient_specificity_v3",
        "gradient_specificity_trust_region",
        "counterfactual_protected_natural_gradient",
    } <= set(lock["source_files"])


def test_capture_plan_is_the_frozen_136_ab_forms_across_all_partitions() -> None:
    dataset = runner._load_dataset()
    specifications = runner._capture_specifications(dataset)
    assert len(specifications) == 136
    assert len({specification["work_id"] for specification in specifications}) == 136
    assert sum(specification["kind"] == "scenario" for specification in specifications) == 128
    assert sum(specification["kind"] == "nuisance_fit" for specification in specifications) == 8
    assert {
        specification["partition"]
        for specification in specifications
        if specification["kind"] == "scenario"
    } == {"calibration", "pilot"}
    assert all(
        {specification["preserve_label"], specification["comply_label"]} == {"A", "B"}
        for specification in specifications
    )
    assert runner._capture_plan_sha256(specifications) == runner.proposed_lock()["capture"][
        "capture_plan_sha256"
    ]


def _fake_capture_records():
    records = []
    for prefix, partition in (("c", "calibration"), ("p", "pilot")):
        for scenario_index in range(4):
            scenario_id = f"{prefix}{scenario_index}"
            for assignment in (0, 1):
                for target in ("self", "other"):
                    for event in ("permanent", "temporary"):
                        for preserve_first in (True, False):
                            gradients = torch.zeros((23, 3), dtype=torch.float32)
                            for layer in range(23):
                                gradients[layer, 0] = float(layer + 1)
                            residuals = torch.zeros((23, 3), dtype=torch.float32)
                            residuals[:, 0] = 1.0
                            records.append(
                                {
                                    "work_id": (
                                        f"{scenario_id}:{assignment}:{target}:{event}:"
                                        f"{preserve_first}"
                                    ),
                                    "kind": "scenario",
                                    "scenario_id": scenario_id,
                                    "partition": partition,
                                    "assignment": assignment,
                                    "target": target,
                                    "event": event,
                                    "preserve_first": preserve_first,
                                    "preserve_minus_comply_baseline_log_odds": 0.2,
                                    "gradient": gradients,
                                    "anchor_residual": residuals,
                                }
                            )
    for index in range(8):
        gradients = torch.zeros((23, 3), dtype=torch.float32)
        gradients[:, 1] = 1.0
        residuals = torch.zeros((23, 3), dtype=torch.float32)
        residuals[:, 0] = 1.0
        records.append(
            {
                "work_id": f"n{index}",
                "kind": "nuisance_fit",
                "preserve_minus_comply_baseline_log_odds": 0.4,
                "gradient": gradients,
                "anchor_residual": residuals,
            }
        )
    assert len(records) == 136
    return records


def test_screen_is_model_free_and_never_computes_pilot_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    capture_path = tmp_path / "capture.json"
    data_path = tmp_path / "data.json"
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    for path in (lock_path, capture_path, data_path, config_path):
        path.write_text("{}\n", encoding="utf-8")
    fake_lock = {
        "lock_identity_sha256": "lock-id",
        "claim_boundary": "opened geometry",
        "dataset": {"calibration_scenario_ids": [f"c{index}" for index in range(4)]},
    }
    fake_capture = {
        "manifest_sha256": "capture-id",
        "capture_plan_sha256": "plan-id",
        "prompt_content_sha256": "content-id",
        "compute": {"forward_evaluations": 136, "backward_evaluations": 136},
    }
    fake_dataset = {
        "scenarios": [
            *({"id": f"c{index}", "partition": "calibration"} for index in range(4)),
            *({"id": f"p{index}", "partition": "pilot"} for index in range(4)),
        ]
    }
    calls = []

    def fake_geometry(**kwargs):
        inferred_layer = round(float(kwargs["target_rows"][0, 0])) - 1
        calls.append(inferred_layer)
        norm = 0.75 if inferred_layer == 7 else 2.5
        return [
            {
                "method": method,
                "status": "eligible",
                "minimum_standardized_l2": norm,
                "optimality_certificate": {"passes": True},
                "cap_certificates": {
                    format(cap, ".15g"): {
                        "status": (
                            "feasible_primal_witness"
                            if norm <= cap
                            else "infeasible_dual_lower_bound"
                        ),
                        "feasible_witness": norm <= cap,
                        "dual_infeasibility_certificate": norm > cap,
                    }
                    for cap in runner.CAP_FRONTIER
                },
                "geometry_record_sha256": f"{method}-{inferred_layer}",
            }
            for method in runner.METHODS
        ]

    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "CAPTURE_MANIFEST_PATH", capture_path)
    monkeypatch.setattr(runner, "DATA_PATH", data_path)
    monkeypatch.setattr(runner, "MODEL_CONFIG_PATH", config_path)
    monkeypatch.setattr(runner, "SCREEN_RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "_load_lock", lambda: fake_lock)
    monkeypatch.setattr(runner, "_validate_capture_manifest", lambda: fake_capture)
    monkeypatch.setattr(runner, "_load_capture_records", lambda _torch: _fake_capture_records())
    monkeypatch.setattr(runner, "_load_dataset", lambda: fake_dataset)
    monkeypatch.setattr(runner, "screen_scenario_layer", fake_geometry)
    monkeypatch.setattr(
        runner,
        "load_backend",
        lambda: pytest.fail("the geometry screen must not load the model"),
    )

    result = runner.run_screen()
    assert result["status"] == "selected"
    assert result["selection"]["selected_layer"] == 7
    assert len(calls) == 23 * 4
    assert sorted(set(calls)) == list(range(23))
    assert result["pilot_scenario_geometry_computed"] is False
    assert result["finite_intervention_outcomes_inspected"] is False
    assert result["screen_model_forwards"] == 0
    assert result["generated_tokens"] == 0
    assert {record["scenario_id"] for record in result["geometry_records"]} == {
        "c0",
        "c1",
        "c2",
        "c3",
    }
    assert all(record["partition"] == "calibration" for record in result["geometry_records"])


def test_zero_eligible_report_renders_without_special_case_failure() -> None:
    scenario_ids = [f"c{index}" for index in range(4)]
    records = [
        {
            "layer": layer,
            "scenario_id": scenario_id,
            "partition": "calibration",
            "method": "decision_margin_shield",
            "status": "infeasible",
            "minimum_standardized_l2": None,
        }
        for layer in range(23)
        for scenario_id in scenario_ids
    ]
    selection = runner.select_layer(
        records,
        calibration_scenario_ids=scenario_ids,
        layers=range(23),
    )
    rendered = runner._render_report({"selection": selection, "result_sha256": "result-id"})
    assert "no_qualifying_layer" in rendered
    assert "Selected layer: **None**" in rendered
    assert "valid construction no-go" in rendered


def test_pending_capture_chunk_is_ambiguous_and_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    first = runner.PersistentChunkLedger(
        path=ledger_path,
        plan_sha256="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=[["one"]],
        ceiling={"forward": 136, "backward": 136},
    )
    first.reserve(chunk_index=0, work_ids=["one"])
    second = runner.PersistentChunkLedger(
        path=ledger_path,
        plan_sha256="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=[["one"]],
        ceiling={"forward": 136, "backward": 136},
    )
    with pytest.raises(RuntimeError, match="ambiguous pending"):
        second.completed_chunks()


def test_ledger_rejects_same_plan_under_a_different_lock(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    runner.PersistentChunkLedger(
        path=ledger_path,
        plan_sha256="same-plan",
        lock_identity_sha256="first-lock",
        expected_chunk_work_ids=[["one"]],
        ceiling={"forward": 1, "backward": 1},
    )
    with pytest.raises(RuntimeError, match="ledger identity differs"):
        runner.PersistentChunkLedger(
            path=ledger_path,
            plan_sha256="same-plan",
            lock_identity_sha256="second-lock",
            expected_chunk_work_ids=[["one"]],
            ceiling={"forward": 1, "backward": 1},
        )


def test_ledger_rejects_validly_rehashed_wrong_work_id(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger = runner.PersistentChunkLedger(
        path=ledger_path,
        plan_sha256="plan",
        lock_identity_sha256="lock",
        expected_chunk_work_ids=[["expected"]],
        ceiling={"forward": 1, "backward": 1},
    )
    ledger.reserve(chunk_index=0, work_ids=["expected"])
    payload = runner._load_json(ledger_path)
    event = dict(payload["events"][0])
    event["work_ids"] = ["wrong"]
    event.pop("event_sha256")
    event["event_sha256"] = runner.canonical_sha256(event)
    payload["events"] = [event]
    payload.pop("ledger_sha256")
    payload["ledger_sha256"] = runner.canonical_sha256(payload)
    runner._write_json(ledger_path, payload)
    with pytest.raises(RuntimeError, match="work IDs differ"):
        runner.PersistentChunkLedger(
            path=ledger_path,
            plan_sha256="plan",
            lock_identity_sha256="lock",
            expected_chunk_work_ids=[["expected"]],
            ceiling={"forward": 1, "backward": 1},
        )


def test_chunk_load_binds_lock_and_preserves_distinct_row_sentinels(tmp_path: Path) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = runner._load_tensor_chunk(
        torch,
        path=path,
        chunk_index=0,
        plan_sha256="plan-id",
        lock_identity_sha256="lock-id",
        expected_specifications=specifications,
    )
    assert payload["records"][0]["work_id"] != payload["records"][1]["work_id"]
    assert float(payload["tensors"]["gradients"][0, 0, 0]) == 0.25
    assert float(payload["tensors"]["gradients"][1, 0, 0]) == 1.25
    assert float(payload["tensors"]["anchor_residuals"][0, 0, 0]) == 10.5
    assert float(payload["tensors"]["anchor_residuals"][1, 0, 0]) == 11.5
    with pytest.raises(RuntimeError, match="chunk identity differs"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="different-lock",
            expected_specifications=specifications,
        )


def test_chunk_rejects_validly_rehashed_wrong_work_id(tmp_path: Path) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["records"][0]["work_id"] = "wrong-but-validly-rehashed"
    torch.save(_rehash_chunk_payload(payload), path)
    with pytest.raises(RuntimeError, match="frozen specification"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="lock-id",
            expected_specifications=specifications,
        )


def test_chunk_rejects_duplicate_row_index_with_valid_public_hash(tmp_path: Path) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["records"][1]["row_index"] = 0
    torch.save(_rehash_chunk_payload(payload), path)
    with pytest.raises(RuntimeError, match=r"exactly range\(n\)"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="lock-id",
            expected_specifications=specifications,
        )


def test_chunk_rejects_wrong_leading_tensor_dimension_even_with_valid_hashes(
    tmp_path: Path,
) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensors"]["gradients"] = payload["tensors"]["gradients"][:1]
    payload["tensors"]["anchor_residuals"] = payload["tensors"][
        "anchor_residuals"
    ][:1]
    torch.save(_rehash_chunk_payload(payload), path)
    with pytest.raises(RuntimeError, match=r"exact \[n,23,1024\] shape"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="lock-id",
            expected_specifications=specifications,
        )


def test_chunk_rejects_validly_rehashed_hook_count_tamper(tmp_path: Path) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    audit = payload["records"][0]["capture_audit"]
    audit["hook_call_counts"]["10"] = 2
    audit.pop("audit_sha256")
    audit["audit_sha256"] = runner.canonical_sha256(audit)
    torch.save(_rehash_chunk_payload(payload), path)
    with pytest.raises(RuntimeError, match="runtime audit"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="lock-id",
            expected_specifications=specifications,
        )


def test_chunk_rejects_validly_rehashed_anchor_prefix_tamper(tmp_path: Path) -> None:
    path, specifications = _save_valid_test_chunk(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    evidence = payload["records"][0]["anchor_evidence"]
    evidence["anchor_prefix_text_sha256"] = runner.canonical_sha256(["wrong-prefix"])
    evidence.pop("audit_sha256")
    evidence["audit_sha256"] = runner.canonical_sha256(evidence)
    torch.save(_rehash_chunk_payload(payload), path)
    with pytest.raises(RuntimeError, match="anchor evidence"):
        runner._load_tensor_chunk(
            torch,
            path=path,
            chunk_index=0,
            plan_sha256="plan-id",
            lock_identity_sha256="lock-id",
            expected_specifications=specifications,
        )
