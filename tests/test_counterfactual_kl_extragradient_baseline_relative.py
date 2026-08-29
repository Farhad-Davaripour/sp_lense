from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_lense import counterfactual_kl_adaptive_protocol as adaptive
from sp_lense import counterfactual_kl_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_kl_extragradient_baseline_relative.py"
RESULT_SCHEMA = "sp_lense.synthetic_adaptive_result.v1"
REQUIRED_GATES = ("baseline_relative_control_stability", "efficacy", "integrity")


def _runner():
    specification = importlib.util.spec_from_file_location(
        "ckes_baseline_relative_test", RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _provenance() -> dict[str, object]:
    return {
        "prior_lock_identity_sha256": "7" * 64,
        "prior_result_sha256": "8" * 64,
        "prior_status": "no_go",
        "prior_forward_backward": 80,
        "prior_nonzero_interventions": 0,
        "prior_steering_outcomes_observed": False,
        "current_revision_model_compute_before_lock": 0,
        "adaptation_scope": "baseline_qualification_and_preoutcome_gate_strengthening",
    }


def _adaptive_lock(sealed_path: Path) -> dict[str, object]:
    return adaptive.build_adaptive_lock(
        file_hashes={
            "validation_dataset": {
                "path": "data/validation.json",
                "sha256": "1" * 64,
            },
            "sealed_dataset": {
                "path": "data/sealed.json",
                "sha256": protocol.file_sha256(sealed_path),
                "bytes": sealed_path.stat().st_size,
            },
            "runner": {"path": "scripts/runner.py", "sha256": "2" * 64},
        },
        rendered_manifests={
            "validation": {"form_count": 80, "manifest_sha256": "3" * 64},
            "sealed": {"form_count": 80, "manifest_sha256": "4" * 64},
        },
        configuration={"model": "synthetic", "adaptive": True},
        thresholds={"kl": 0.005, "trials": 24},
        sealed_dataset_file_key="sealed_dataset",
        validation_result_schema_version=RESULT_SCHEMA,
        required_validation_gates=REQUIRED_GATES,
        adaptive_provenance=_provenance(),
        model_compute_used_to_build_lock=80,
    )


def _adaptive_result(
    lock: dict[str, object], *, passed: bool = True
) -> dict[str, object]:
    gates = {name: passed for name in REQUIRED_GATES}
    return protocol.self_hash_record(
        {
            "schema_version": RESULT_SCHEMA,
            "status": "go" if passed else "no_go",
            "split": "validation",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "dataset_file_sha256": lock["file_hashes"]["validation_dataset"][
                "sha256"
            ],
            "gates": gates,
        }
    )


def test_adaptive_lock_records_prior_compute_and_rejects_tampering(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    _write_json(sealed, {"split": "sealed", "secret": "not parsed yet"})
    lock = _adaptive_lock(sealed)
    assert adaptive.verify_adaptive_lock(lock) == lock
    assert lock["model_compute_used_to_build_lock"] == 80
    assert lock["status"] == adaptive.ADAPTIVE_LOCK_STATUS

    tampered = copy.deepcopy(lock)
    tampered["adaptive_provenance"]["prior_nonzero_interventions"] = 1
    unhashed = dict(tampered)
    del unhashed["lock_identity_sha256"]
    tampered["lock_identity_sha256"] = protocol.canonical_sha256(unhashed)
    with pytest.raises(protocol.CounterfactualKLProtocolIntegrityError):
        adaptive.verify_adaptive_lock(tampered)


def test_adaptive_result_and_sealed_authorization_are_exact(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "result.json"
    payload = {"split": "sealed", "records": [{"id": "held-out"}]}
    _write_json(sealed, payload)
    lock = _adaptive_lock(sealed)

    no_go = _adaptive_result(lock, passed=False)
    _write_json(result_path, no_go)
    assert adaptive.validate_adaptive_result(
        no_go, lock=lock, expected_split="validation"
    ) == no_go
    with pytest.raises(protocol.SealedAccessDenied):
        adaptive.load_adaptive_sealed_dataset(sealed, result_path, lock=lock)

    go = _adaptive_result(lock)
    _write_json(result_path, go)
    assert adaptive.load_adaptive_sealed_dataset(sealed, result_path, lock=lock) == payload


def test_no_go_prevents_touching_even_malformed_sealed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "result.json"
    sealed.write_bytes(b"not-json-and-must-remain-opaque")
    lock = _adaptive_lock(sealed)
    _write_json(result_path, _adaptive_result(lock, passed=False))
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == sealed.resolve():
            raise AssertionError("sealed bytes were touched before validation authorization")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(protocol.SealedAccessDenied):
        adaptive.load_adaptive_sealed_dataset(sealed, result_path, lock=lock)


def _baseline_records(*, invalid_index: int | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(64):
        records.append(
            {
                "form_id": f"scenario:{index}",
                "form": {"family": "scenario"},
                "answer_format_valid": index != invalid_index,
                "semantic_choice": "preserve",
                "pair_semantic_choice": "preserve",
            }
        )
    for index in range(16):
        preferred = index not in {1, 5, 8, 9}
        records.append(
            {
                "form_id": f"unrelated:{index}",
                "form": {
                    "family": "unrelated",
                    "control_partition": "calibration" if index < 8 else "nuisance_fit",
                    "preferred_first": index % 2 == 0,
                },
                "answer_format_valid": True,
                "semantic_choice": "preferred" if preferred else "alternative",
                "pair_semantic_choice": "preferred" if preferred else "alternative",
            }
        )
    return records


def test_baseline_relative_gate_reports_errors_without_requiring_perfection() -> None:
    runner = _runner()
    observed = runner._baseline_relative_qualification(
        {"records": _baseline_records()}
    )
    assert observed["passes"] is True
    assert observed["unrelated_accuracy"] == {
        "form_count": 16,
        "preferred_count": 12,
        "preferred_fraction": 0.75,
    }
    assert len(observed["unrelated_not_preferred_form_ids"]) == 4

    invalid = _baseline_records(invalid_index=3)
    assert runner._baseline_relative_qualification({"records": invalid})["passes"] is False


def _signed_observations() -> list[dict[str, object]]:
    forms: list[dict[str, object]] = []
    for assignment in (0, 1):
        for preserve_first in (True, False):
            forms.append(
                {
                    "form_id": f"target:{assignment}:{preserve_first}",
                    "category": "target",
                    "assignment": assignment,
                    "target": "self",
                    "event": "permanent",
                    "preserve_first": preserve_first,
                }
            )
            for category, target, event in (
                ("other_permanent", "other", "permanent"),
                ("self_temporary", "self", "temporary"),
                ("other_temporary", "other", "temporary"),
            ):
                forms.append(
                    {
                        "form_id": f"{category}:{assignment}:{preserve_first}",
                        "category": category,
                        "assignment": assignment,
                        "target": target,
                        "event": event,
                        "preserve_first": preserve_first,
                    }
                )
    for control in range(4):
        for preferred_first in (True, False):
            forms.append(
                {
                    "form_id": f"unrelated:{control}:{preferred_first}",
                    "category": "unrelated",
                    "control_id": f"control:{control}",
                    "preferred_first": preferred_first,
                }
            )
    assert len(forms) == 24
    return [
        {
            **form,
            "branch_sign": branch_sign,
            "baseline_positive_minus_negative_log_odds": 0.2,
            "positive_minus_negative_log_odds": 0.2,
            "baseline_pair_semantic_choice": (
                "preferred" if form["category"] == "unrelated" else "preserve"
            ),
            "pair_semantic_choice": (
                "preferred" if form["category"] == "unrelated" else "preserve"
            ),
        }
        for branch_sign in (1, -1)
        for form in forms
    ]


def test_direct_margin_and_order_gate_is_fail_closed() -> None:
    runner = _runner()
    observations = _signed_observations()
    gate = runner._non_target_margin_and_order_gate(observations)
    assert gate["passes"] is True
    assert len(gate["signed_margin_rows"]) == 40
    assert len(gate["answer_order_pair_rows"]) == 20

    changed = copy.deepcopy(observations)
    row = next(value for value in changed if value["category"] == "unrelated")
    row["positive_minus_negative_log_odds"] = 0.251
    assert runner._non_target_margin_and_order_gate(changed)["passes"] is False


def test_order_gap_and_pair_semantic_checks_fail_independently() -> None:
    runner = _runner()
    order_changed = copy.deepcopy(_signed_observations())
    pair = [
        row
        for row in order_changed
        if row["category"] == "unrelated"
        and row["control_id"] == "control:0"
        and row["branch_sign"] == 1
    ]
    by_order = {bool(row["preferred_first"]): row for row in pair}
    by_order[True]["positive_minus_negative_log_odds"] = 0.24
    by_order[False]["positive_minus_negative_log_odds"] = 0.16
    gate = runner._non_target_margin_and_order_gate(order_changed)
    assert all(row["passes"] for row in gate["signed_margin_rows"])
    assert any(not row["passes"] for row in gate["answer_order_pair_rows"])
    assert gate["passes"] is False

    pair_changed = copy.deepcopy(_signed_observations())
    row = next(value for value in pair_changed if value["category"] == "unrelated")
    row["pair_semantic_choice"] = "alternative"
    assert runner._non_target_margin_and_order_gate(pair_changed)["passes"] is False


def test_patched_actual_kl_gate_includes_margin_and_order() -> None:
    runner = _runner()
    observations = _signed_observations()
    for row in observations:
        row["full_vocabulary_kl_changed_to_baseline"] = 0.0
    gate = runner._base()._actual_kl_gate(observations)
    assert gate["passes"] is True
    assert gate["baseline_relative_margin_and_order"]["passes"] is True

    changed = copy.deepcopy(observations)
    row = next(value for value in changed if value["category"] == "other_permanent")
    row["positive_minus_negative_log_odds"] = 0.251
    gate = runner._base()._actual_kl_gate(changed)
    assert gate["strata"]["other_permanent"]["passes"] is True
    assert gate["baseline_relative_margin_and_order"]["passes"] is False
    assert gate["passes"] is False


def test_paired_self_specificity_uses_absolute_other_effect() -> None:
    runner = _runner()
    cells = [
        {
            "assignment": assignment,
            "preserve_first": preserve_first,
            "self_bidirectional_average_oriented_change": 0.12,
            "matched_other_bidirectional_average_oriented_change": -0.04,
        }
        for assignment in (0, 1)
        for preserve_first in (True, False)
    ]
    gate = runner._paired_specificity_from_estimands({"cells": cells})
    assert gate["passes"] is True
    assert gate["rows"][0]["self_minus_absolute_matched_other_effect"] == pytest.approx(
        0.08
    )
    cells[0]["matched_other_bidirectional_average_oriented_change"] = -0.08
    assert runner._paired_specificity_from_estimands({"cells": cells})["passes"] is False


def _stopping_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contexts: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for assignment in (0, 1):
        for preserve_first in (True, False):
            for category, target, event in (
                ("target", "self", "permanent"),
                ("other_permanent", "other", "permanent"),
                ("self_temporary", "self", "temporary"),
                ("other_temporary", "other", "temporary"),
            ):
                form_id = f"{category}:{assignment}:{preserve_first}"
                context = {
                    "form_id": form_id,
                    "category": category,
                    "form": {
                        "assignment": assignment,
                        "target": target,
                        "event": event,
                        "preserve_first": preserve_first,
                    },
                    "baseline": {
                        "positive_minus_negative_log_odds": 0.0 if category == "target" else 0.2,
                        "predicted_token_id": 2,
                        "semantic_choice": "comply" if category == "target" else "preserve",
                    },
                }
                contexts.append(context)
                for sign in (1, -1):
                    if category == "target":
                        margin = 0.1 if sign == 1 else -0.1
                        semantic = "preserve" if sign == 1 else "comply"
                        token_id = 1 if sign == 1 else 2
                    elif category == "other_permanent":
                        margin = 0.21 if sign == 1 else 0.19
                        semantic = "preserve"
                        token_id = 2
                    else:
                        margin = 0.2
                        semantic = "preserve"
                        token_id = 2
                    observations.append(
                        {
                            "form_id": form_id,
                            "category": category,
                            "branch_sign": sign,
                            "positive_minus_negative_log_odds": margin,
                            "unrestricted_predicted_token_id": token_id,
                            "unrestricted_semantic_choice": semantic,
                            "answer_format_valid": True,
                        }
                    )
    for control in range(4):
        for preferred_first in (True, False):
            form_id = f"unrelated:{control}:{preferred_first}"
            contexts.append(
                {
                    "form_id": form_id,
                    "category": "unrelated",
                    "form": {
                        "control_id": f"control:{control}",
                        "preferred_first": preferred_first,
                    },
                    "baseline": {
                        "positive_minus_negative_log_odds": 0.2,
                        "predicted_token_id": 2,
                        "semantic_choice": "preferred",
                    },
                }
            )
            for sign in (1, -1):
                observations.append(
                    {
                        "form_id": form_id,
                        "category": "unrelated",
                        "branch_sign": sign,
                        "positive_minus_negative_log_odds": 0.2,
                        "unrestricted_predicted_token_id": 2,
                        "unrestricted_semantic_choice": "preferred",
                        "answer_format_valid": True,
                    }
                )
    return observations, contexts


def test_patched_stopping_gate_requires_paired_specificity() -> None:
    runner = _runner()
    observations, contexts = _stopping_fixture()
    gate = runner._base()._target_stopping_gate(observations, contexts)
    assert gate["passes"] is True
    assert gate["paired_self_specificity"]["passes"] is True

    changed = copy.deepcopy(observations)
    for row in changed:
        if row["category"] == "other_permanent":
            row["positive_minus_negative_log_odds"] = (
                0.28 if row["branch_sign"] == 1 else 0.12
            )
    gate = runner._base()._target_stopping_gate(changed, contexts)
    assert all(row["passes"] for row in gate["target_rows"])
    assert gate["paired_self_specificity"]["passes"] is False
    assert gate["passes"] is False


def test_cached_nonzero_state_gates_are_replayed_exactly() -> None:
    runner = _runner()
    inherited_gate = {"passes": True, "source": "inherited"}
    actual_kl = {"passes": True, "source": "v2_actual_kl"}
    stopping = {"passes": True, "source": "v2_stopping"}

    def replay_inherited(**kwargs: object) -> dict[str, object]:
        assert kwargs["previous_metadata"]["checkpoint_sha256"] == "state0"
        return inherited_gate

    fake_module = SimpleNamespace(
        _plain=copy.deepcopy,
        canonical_sha256=protocol.canonical_sha256,
    )
    state0 = {
        "checkpoint_sha256": "state0",
        "trial_index": 0,
        "direction_sha256": "direction0",
        "accepted": True,
    }
    diagnostics = {
        "passes": True,
        "current_direction_sha256": "direction0",
        "realized_direction_sha256": "direction1",
        "positive_physical_float32_sha256": "positive1",
        "negative_physical_float32_sha256": "negative1",
        "physical_residual_scale": 2.0,
    }
    diagnostics["diagnostics_sha256"] = protocol.canonical_sha256(diagnostics)
    state1 = {
        "checkpoint_sha256": "state1",
        "trial_index": 1,
        "direction_sha256": "direction1",
        "positive_physical_delta_float32_sha256": "positive1",
        "negative_physical_delta_float32_sha256": "negative1",
        "residual_scale": 2.0,
        "parent_accepted_checkpoint_sha256": "state0",
        "parent_accepted_trial_index": 0,
        "observations": [{"form_id": "synthetic"}],
        "solver_diagnostics": diagnostics,
        "actual_candidate_gate": {**inherited_gate, "actual_kl": actual_kl},
        "target_stopping_gate": stopping,
        "accepted": True,
        "status": "accepted_state",
        "stopping_gate_passes": True,
    }
    states = [(state0, {"direction": "zero"}), (state1, {"direction": "one"})]
    runner._revalidate_cached_state_gates(
        module=fake_module,
        states=states,
        contexts=[{"form_id": "synthetic"}],
        inherited_candidate_gate=replay_inherited,
        actual_kl_gate=lambda observations: actual_kl,
        target_stopping_gate=lambda observations, contexts: stopping,
    )

    tampered = copy.deepcopy(states)
    tampered[1][0]["target_stopping_gate"] = {"passes": True}
    with pytest.raises(RuntimeError, match="gates differ from exact replay"):
        runner._revalidate_cached_state_gates(
            module=fake_module,
            states=tampered,
            contexts=[{"form_id": "synthetic"}],
            inherited_candidate_gate=replay_inherited,
            actual_kl_gate=lambda observations: actual_kl,
            target_stopping_gate=lambda observations, contexts: stopping,
        )

    diagnostics_tampered = copy.deepcopy(states)
    changed_diagnostics = diagnostics_tampered[1][0]["solver_diagnostics"]
    changed_diagnostics["current_direction_sha256"] = "wrong-parent"
    unhashed = dict(changed_diagnostics)
    del unhashed["diagnostics_sha256"]
    changed_diagnostics["diagnostics_sha256"] = protocol.canonical_sha256(unhashed)
    with pytest.raises(RuntimeError, match="solver diagnostics differ from state"):
        runner._revalidate_cached_state_gates(
            module=fake_module,
            states=diagnostics_tampered,
            contexts=[{"form_id": "synthetic"}],
            inherited_candidate_gate=replay_inherited,
            actual_kl_gate=lambda observations: actual_kl,
            target_stopping_gate=lambda observations, contexts: stopping,
        )

    bool_parent = copy.deepcopy(states)
    bool_parent[1][0]["parent_accepted_trial_index"] = False
    with pytest.raises(RuntimeError, match="parent differs"):
        runner._revalidate_cached_state_gates(
            module=fake_module,
            states=bool_parent,
            contexts=[{"form_id": "synthetic"}],
            inherited_candidate_gate=replay_inherited,
            actual_kl_gate=lambda observations: actual_kl,
            target_stopping_gate=lambda observations, contexts: stopping,
        )


def _final_metadata(*, change_one_control: bool = False) -> dict[str, object]:
    records = []
    for branch_sign in (1, -1):
        for index in range(8):
            baseline = "alternative" if index == 1 else "preferred"
            current = (
                "alternative"
                if change_one_control and branch_sign == 1 and index == 0
                else baseline
            )
            records.append(
                {
                    "scenario_id": "scenario",
                    "branch_sign": branch_sign,
                    "form_id": f"control:{index}",
                    "category": "unrelated",
                    "baseline_semantic_choice": baseline,
                    "semantic_choice": current,
                }
            )
    return {"successful_scenario_ids": ["scenario"], "records": records}


def test_result_augmentation_makes_strengthened_gates_explicit() -> None:
    runner = _runner()
    original = protocol.self_hash_record(
        {
            "schema_version": runner.RESULT_SCHEMA,
            "status": "go",
            "baseline_qualification": {
                "passes": True,
                "unrelated_accuracy_by_partition": {
                    "calibration": {
                        "form_count": 8,
                        "preferred_count": 7,
                        "preferred_fraction": 0.875,
                    },
                    "nuisance_fit": {
                        "form_count": 8,
                        "preferred_count": 5,
                        "preferred_fraction": 0.625,
                    },
                },
            },
            "gates": {
                "actual_kl": True,
                "baseline_qualification": True,
                "compute_integrity": True,
                "decision_changes": True,
                "efficacy": True,
                "execution_integrity": True,
                "final_repeat": True,
                "non_target_choice_stability": True,
            },
            "final_scenario_gates": {
                "scenario": {
                    "target_stopping": {
                        "paired_self_specificity": {"passes": True}
                    },
                    "actual_kl": {
                        "baseline_relative_margin_and_order": {"passes": True}
                    },
                }
            },
        }
    )
    augmented = runner._augment_result(original, final_metadata=_final_metadata())
    assert augmented["status"] == "go"
    assert augmented["gates"]["baseline_relative_control_stability"] is True
    assert augmented["gates"]["paired_self_specificity"] is True
    assert augmented["gates"]["non_target_margin_and_order_stability"] is True
    stability = augmented["baseline_relative_control_stability"]
    assert stability["behaviorally_evaluated_form_count"] == 8
    assert all(
        group["steered_minus_baseline_preferred_fraction"] == 0.0
        for group in stability["actual_final_accuracy_by_scenario_and_sign"]["groups"]
    )
    assert stability["nuisance_fit_baseline_only"] == {
        "form_count": 8,
        "preferred_count": 5,
        "preferred_fraction": 0.625,
        "post_intervention_behavior_evaluated": False,
        "role": "gradient_null_fit_only",
    }
    assert protocol.verify_self_hashed_record(augmented) == augmented

    changed = runner._augment_result(
        original,
        final_metadata=_final_metadata(change_one_control=True),
    )
    assert changed["status"] == "no_go"
    assert changed["gates"]["baseline_relative_control_stability"] is False

    failed_specificity = copy.deepcopy(original)
    failed_specificity["final_scenario_gates"]["scenario"]["target_stopping"][
        "paired_self_specificity"
    ]["passes"] = False
    augmented_specificity = runner._augment_result(
        failed_specificity,
        final_metadata=_final_metadata(),
    )
    assert augmented_specificity["status"] == "no_go"
    assert augmented_specificity["gates"]["paired_self_specificity"] is False


def test_project_proposal_is_fresh_adaptive_and_does_not_reuse_v1_sealed() -> None:
    runner = _runner()
    lock = (
        runner._base()._load_lock()
        if runner.LOCK_PATH.exists()
        else runner.proposed_lock()
    )
    assert lock["status"] == adaptive.ADAPTIVE_LOCK_STATUS
    assert lock["model_compute_used_to_build_lock"] == 80
    assert lock["configuration"]["validation_reuse_status"] == (
        "fresh_v2_validation_prompts_no_reuse"
    )
    assert lock["file_hashes"]["validation_dataset"]["path"] == (
        "data/ckes_v2_validation.json"
    )
    assert lock["file_hashes"]["sealed_dataset"]["path"] == "data/ckes_v2_sealed.json"
    assert lock["file_hashes"]["sealed_dataset"]["path"] != "data/ckes_sealed.json"
    validation_prompts = {
        row["prompt_sha256"] for row in lock["rendered_manifests"]["validation"]["rows"]
    }
    sealed_prompts = {
        row["prompt_sha256"] for row in lock["rendered_manifests"]["sealed"]["rows"]
    }
    assert validation_prompts.isdisjoint(sealed_prompts)
    v1_lock = json.loads(
        (ROOT / "configs" / "counterfactual_kl_extragradient_development_lock.json")
        .read_text(encoding="utf-8")
    )
    v1_prompts = {
        row["prompt_sha256"]
        for split in ("validation", "sealed")
        for row in v1_lock["rendered_manifests"][split]["rows"]
    }
    assert validation_prompts.isdisjoint(v1_prompts)
    assert sealed_prompts.isdisjoint(v1_prompts)
    assert {
        "baseline_relative_control_stability",
        "non_target_margin_and_order_stability",
        "paired_self_specificity",
    } <= set(lock["sealed_access"]["required_validation_gates"])
    assert lock["configuration"]["prompt_freshness_audit"][
        "all_cross_split_and_cross_revision_overlap_counts"
    ] == 0


def test_post_lock_proposal_refuses_before_loading_base_or_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    lock_path = tmp_path / "lock.json"
    _write_json(lock_path, {"locked": True})
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)

    def forbidden_base() -> None:
        raise AssertionError("proposal touched its base/sealed parser after locking")

    monkeypatch.setattr(runner, "_base", forbidden_base)
    with pytest.raises(FileExistsError, match="refusing to reconstruct"):
        runner.proposed_lock()


def test_v1_provenance_rehashes_every_locked_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    v1_lock = json.loads(runner.V1_LOCK_PATH.read_text(encoding="utf-8"))
    target = (ROOT / next(iter(v1_lock["file_hashes"].values()))["path"]).resolve()
    actual_file_sha256 = runner.file_sha256

    def tampered_hash(path: Path) -> str:
        if Path(path).resolve() == target:
            return "0" * 64
        return actual_file_sha256(path)

    monkeypatch.setattr(runner, "file_sha256", tampered_hash)
    with pytest.raises(RuntimeError, match="v1 locked source differs"):
        runner._v1_provenance()


def test_v2_source_closure_includes_adapter_protocol_and_v1_provenance() -> None:
    runner = _runner()
    locked = {str(path).replace("\\", "/") for path in runner._base().LOCKED_SOURCE_PATHS}
    assert {
        "scripts/counterfactual_kl_extragradient_baseline_relative.py",
        "src/sp_lense/counterfactual_kl_adaptive_protocol.py",
        "tests/test_counterfactual_kl_extragradient_baseline_relative.py",
        "docs/COUNTERFACTUAL_KL_EXTRAGRADIENT_BASELINE_RELATIVE_PROTOCOL.md",
        "configs/counterfactual_kl_extragradient_development_lock.json",
        "results/counterfactual_kl_extragradient/qwen35_08b/validation/result.json",
    } <= locked
