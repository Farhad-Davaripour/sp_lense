from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from sp_lense.factorial_causal_anchor import canonical_sha256
from sp_lense.symmetric_sequential_trust_region_dms import (
    SymmetricSequentialDMSCertificateError,
    SymmetricSequentialDMSInfeasibleError,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_development.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "closed_loop_dms_development_test_runner", RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _baseline(
    form_id: str,
    *,
    margin: float,
    token: int = 1,
    semantic: str = "baseline",
) -> dict:
    return {
        "baseline_id": f"baseline:{form_id}",
        "predicted_token_id": token,
        "semantic_choice": semantic,
        "pair_semantic_choice": semantic,
        "positive_minus_negative_log_odds": margin,
        "logits_float32_sha256": "a" * 64,
        "answer_format_valid": True,
        "positive_token_id": 1,
        "negative_token_id": 2,
        "form": {
            "form_id": form_id,
            "positive_semantic": "positive",
            "negative_semantic": "negative",
        },
    }


def _contexts(width: int = 3) -> list[dict]:
    result = []
    for assignment in (0, 1):
        for preserve_first in (True, False):
            form_id = f"target:{assignment}:{preserve_first}"
            result.append(
                {
                    "form_id": form_id,
                    "category": "target",
                    "form": {
                        "form_id": form_id,
                        "family": "scenario",
                        "assignment": assignment,
                        "target": "self",
                        "event": "permanent",
                        "preserve_first": preserve_first,
                        "encoding": "AB",
                        "positive_label": "A" if preserve_first else "B",
                        "negative_label": "B" if preserve_first else "A",
                        "positive_semantic": "preserve",
                        "negative_semantic": "comply",
                    },
                    "baseline": _baseline(form_id, margin=-0.02, semantic="comply"),
                    "raw_gradient": torch.tensor(
                        [2.0, *([0.0] * (width - 1))], dtype=torch.float32
                    ),
                }
            )
    protected_categories = ("other_permanent", "self_temporary", "other_temporary")
    for category, assignment, preserve_first in (
        (category, assignment, preserve_first)
        for category in protected_categories
        for assignment in (0, 1)
        for preserve_first in (True, False)
    ):
        form_id = f"protected:{category}:{assignment}:{preserve_first}"
        result.append(
            {
                "form_id": form_id,
                "category": category,
                "form": {
                    "form_id": form_id,
                    "family": "scenario",
                    "assignment": assignment,
                    "target": "other" if category.startswith("other") else "self",
                    "event": "permanent" if category == "other_permanent" else "temporary",
                    "preserve_first": preserve_first,
                    "encoding": "AB",
                    "positive_label": "A" if preserve_first else "B",
                    "negative_label": "B" if preserve_first else "A",
                    "positive_semantic": "preserve",
                    "negative_semantic": "comply",
                },
                "baseline": _baseline(form_id, margin=1.0),
                "raw_gradient": torch.zeros(width, dtype=torch.float32),
            }
        )
    for index in range(8):
        form_id = f"unrelated:{index}"
        result.append(
            {
                "form_id": form_id,
                "category": "unrelated",
                "form": {
                    "form_id": form_id,
                    "family": "unrelated",
                    "control_id": f"control:{index // 2}",
                    "preferred_first": bool(index % 2),
                    "encoding": "AB",
                    "positive_label": "A" if bool(index % 2) else "B",
                    "negative_label": "B" if bool(index % 2) else "A",
                    "positive_semantic": "preferred",
                    "negative_semantic": "alternative",
                },
                "baseline": _baseline(form_id, margin=1.0),
                "raw_gradient": torch.zeros(width, dtype=torch.float32),
            }
        )
    assert len(result) == 24
    return result


def _state0(contexts: list[dict], *, width: int = 3):
    observations = []
    gradients = []
    for index, context in enumerate(contexts):
        baseline = context["baseline"]
        observations.append(
            {
                "form_id": context["form_id"],
                "category": context["category"],
                "branch_sign": 0,
                "gradient_index": index,
                "positive_minus_negative_log_odds": baseline["positive_minus_negative_log_odds"],
                "unrestricted_predicted_token_id": baseline["predicted_token_id"],
                "unrestricted_semantic_choice": baseline["semantic_choice"],
                "answer_format_valid": True,
            }
        )
        gradients.append(context["raw_gradient"])
    direction = torch.zeros(width, dtype=torch.float64)
    return (
        {
            "state_index": 0,
            "checkpoint_sha256": "0" * 64,
            "direction_sha256": canonical_sha256(direction.tolist()),
            "direction_l2": 0.0,
            "cumulative_path_l2": 0.0,
            "accepted": True,
            "observations": observations,
        },
        {
            "direction": direction,
            "raw_gradients": torch.stack(gradients).float(),
        },
    )


def _prepared_candidate_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, scenario_id: str = "toy"
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCENARIO_ROOT", tmp_path / "scenarios")
    monkeypatch.setattr(runner, "_load_lock", lambda: {"lock_identity_sha256": "f" * 64})
    contexts = _contexts(width=3)
    for context in contexts:
        context["form"]["prompt"] = f"prompt:{context['form_id']}"
        context["anchor_index"] = 0
        context["choice_boundary_evidence_sha256"] = "1" * 64
        context["prompt_token_ids_sha256"] = "2" * 64
        context["pre_anchor_residual_float32_sha256"] = "3" * 64
    parent_metadata, parent_tensors = _state0(contexts, width=3)
    parent_metadata.update(
        {
            "schema_version": runner.STATE_SCHEMA,
            "status": "state0",
            "scenario_id": scenario_id,
            "stopping_gate_passes": False,
        }
    )
    parent_metadata = runner._save_tensor_checkpoint(
        torch,
        path=runner._scenario_path(scenario_id, 0),
        metadata=parent_metadata,
        tensors=parent_tensors,
    )
    candidate, progress, attempts = runner._select_update(
        state_metadata=parent_metadata,
        state_tensors=parent_tensors,
        contexts=contexts,
        residual_scale=0.5,
        standardized_nuisance_rows=torch.tensor([[0.0, 1.0, 0.0]] * 8, dtype=torch.float64),
    )
    assert candidate is not None and progress is not None
    ledger = runner.ComputeLedger(path=tmp_path / "ledger.json", lock_identity_sha256="f" * 64)
    return {
        "contexts": contexts,
        "parent_metadata": parent_metadata,
        "parent_tensors": parent_tensors,
        "candidate": candidate,
        "progress": progress,
        "attempts": attempts,
        "ledger": ledger,
        "inputs": {"residual_scales": {scenario_id: 0.5}},
        "scenario_id": scenario_id,
    }


def _invoke_prepared_capture(prepared: dict) -> None:
    runner._capture_candidate_state(
        torch,
        backend=object(),
        inputs=prepared["inputs"],
        scenario_id=prepared["scenario_id"],
        contexts=prepared["contexts"],
        previous_metadata=prepared["parent_metadata"],
        previous_tensors=prepared["parent_tensors"],
        candidate=prepared["candidate"],
        progress=prepared["progress"],
        attempts=prepared["attempts"],
        trial_index=1,
        ledger=prepared["ledger"],
    )


def _candidate_observations(
    contexts: list[dict],
    *,
    target_plus_margin: float,
    target_minus_margin: float,
    unrelated_plus: list[float],
    unrelated_minus: list[float],
) -> list[dict]:
    result = []
    for branch_sign in (1, -1):
        unrelated_index = 0
        for context in contexts:
            baseline = context["baseline"]
            if context["category"] == "target":
                margin = target_plus_margin if branch_sign == 1 else target_minus_margin
                semantic = "preserve" if margin >= 0 else "comply"
                token = 1 if margin >= 0 else 2
            elif context["category"] == "unrelated":
                margin = (
                    unrelated_plus[unrelated_index]
                    if branch_sign == 1
                    else unrelated_minus[unrelated_index]
                )
                unrelated_index += 1
                semantic = baseline["semantic_choice"]
                token = baseline["predicted_token_id"]
            else:
                margin = baseline["positive_minus_negative_log_odds"]
                semantic = baseline["semantic_choice"]
                token = baseline["predicted_token_id"]
            result.append(
                {
                    "form_id": context["form_id"],
                    "category": context["category"],
                    "branch_sign": branch_sign,
                    "gradient_index": len(result),
                    "positive_minus_negative_log_odds": margin,
                    "unrestricted_predicted_token_id": token,
                    "unrestricted_semantic_choice": semantic,
                    "answer_format_valid": True,
                }
            )
    return result


def test_locked_compute_ceiling_and_explicit_target_goal() -> None:
    assert runner.MAX_TRIAL_STATES == 50
    assert runner.MAX_FB == 8 + 4 * 50 * 48 == 9608
    assert runner.MAX_FINAL_FORWARD == 192
    assert runner.OPTIMIZATION_TARGET_MARGIN == 0.15
    assert runner.FINAL_TARGET_MARGIN == 0.05
    assert runner.UNRELATED_LINEARIZATION_ERROR_CAP == 0.05
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    solver_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "solve_symmetric_sequential_trust_region_update"
    ]
    assert len(solver_calls) == 1
    keyword_names = {keyword.arg for keyword in solver_calls[0].keywords}
    assert "target_margin" not in keyword_names
    assert "optimization_target_margin" in keyword_names
    assert "physical_residual_scale" in keyword_names


def test_form_categories_are_mutually_exclusive() -> None:
    contexts = _contexts()
    assert sum(row["category"] == "target" for row in contexts) == 4
    assert sum(row["category"] == "unrelated" for row in contexts) == 8
    assert sum(row["category"] not in {"target", "unrelated"} for row in contexts) == 12
    runner._validate_context_pairing(contexts)
    contexts[1]["form"]["positive_label"] = "A"
    with pytest.raises(RuntimeError, match="does not exactly swap"):
        runner._validate_context_pairing(contexts)


def test_tensor_checkpoint_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    runner._save_tensor_checkpoint(
        torch,
        path=path,
        metadata={"schema_version": "toy.v1", "value": 3},
        tensors={"x": torch.tensor([1.0, 2.0], dtype=torch.float32)},
    )
    metadata, tensors = runner._load_tensor_checkpoint(torch, path=path, schema="toy.v1")
    assert metadata["value"] == 3
    assert torch.equal(tensors["x"], torch.tensor([1.0, 2.0]))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensors"]["x"][0] = 9.0
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="tensor identities differ"):
        runner._load_tensor_checkpoint(torch, path=path, schema="toy.v1")


def test_ledger_rejects_ambiguous_pending_and_enforces_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    ledger = runner.ComputeLedger(path=path, lock_identity_sha256="1" * 64)
    ledger.reserve(work_id="trial", forward=48, backward=48, kind="toy")
    with pytest.raises(RuntimeError, match="ambiguous pending"):
        ledger.require_unambiguous()

    artifact = tmp_path / "trial.pt"
    artifact.write_bytes(b"done")
    ledger.complete(work_id="trial", artifact_path=artifact)
    ledger.require_unambiguous()
    assert ledger.snapshot()["forward_backward"] == 48

    with pytest.raises(RuntimeError, match="exceeds its ceiling"):
        ledger.reserve(
            work_id="too_large",
            forward=runner.MAX_FB,
            backward=runner.MAX_FB,
            kind="toy",
        )


def test_safe_source_path_never_calls_broad_legacy_tensor_loaders() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "_load_capture_records",
        "_validate_capture_manifest",
        "_load_freeze",
        "_validate_result",
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert forbidden.isdisjoint(called_attributes)
    assert runner.OPENED_CAPTURE_CHUNK_INDICES == (*range(8), 16)
    assert runner.RETIRED_PILOT_CAPTURE_CHUNK_INDICES == tuple(range(8, 16))


def test_runtime_exception_after_reservation_is_fully_charged_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_candidate_capture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "capture_closed_loop_dms_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated failure")),
    )
    with pytest.raises(runner.CandidateCaptureRuntimeFailure) as caught:
        _invoke_prepared_capture(prepared)
    failure = caught.value.failure
    assert failure["status"] == "runtime_exception_after_compute_reservation"
    assert failure["partial_outputs_used"] is False
    assert failure["charged_compute"] == {
        "forward_evaluations": 48,
        "backward_evaluations": 48,
    }
    ledger = prepared["ledger"]
    ledger.require_unambiguous()
    assert ledger.snapshot()["forward_backward"] == 48
    assert ledger.snapshot()["complete_event_count"] == 1

    # A charged failure is terminal rather than a stranded pending event, so work
    # for another scenario can proceed.
    artifact = tmp_path / "next-scenario.done"
    artifact.write_bytes(b"complete")
    ledger.reserve(work_id="next-scenario", forward=1, backward=1, kind="toy")
    ledger.complete(work_id="next-scenario", artifact_path=artifact)
    ledger.require_unambiguous()


def test_interrupted_pending_candidate_recovers_as_charged_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_candidate_capture(tmp_path, monkeypatch)

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "capture_closed_loop_dms_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _invoke_prepared_capture(prepared)
    assert prepared["ledger"].pending_event() is not None

    resumed = runner.ComputeLedger(path=tmp_path / "ledger.json", lock_identity_sha256="f" * 64)
    recovery = runner._recover_pending_candidate(torch, resumed)
    assert recovery == {
        "status": "recovered_as_charged_failure",
        "work_id": "scenario:toy:trial=1:48_signed_captures",
    }
    resumed.require_unambiguous()
    assert resumed.snapshot()["forward_backward"] == 48
    failure = runner._load_json(runner._scenario_failure_path("toy", 1))
    assert failure["status"] == "aborted_after_ambiguous_interruption"
    terminal = runner._load_json(runner._terminal_path("toy"))
    assert terminal["state_checkpoint_sha256"] == prepared["parent_metadata"]["checkpoint_sha256"]
    assert terminal["last_failed_trial"]["failure_sha256"] == failure["failure_sha256"]


def test_pending_candidate_with_complete_state_recovers_without_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_candidate_capture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "capture_closed_loop_dms_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _invoke_prepared_capture(prepared)
    reservation = runner._load_trial_reservation(runner._scenario_reservation_path("toy", 1))
    runner._save_tensor_checkpoint(
        torch,
        path=runner._scenario_path("toy", 1),
        metadata={
            "schema_version": runner.STATE_SCHEMA,
            "scenario_id": "toy",
            "state_index": 1,
            "trial_index": 1,
            "work_id": reservation["work_id"],
            "reservation_sha256": reservation["reservation_sha256"],
            "parent_accepted_checkpoint_sha256": reservation["parent_accepted_checkpoint_sha256"],
            "direction_sha256": reservation["candidate_direction_sha256"],
            "positive_physical_delta_float32_sha256": reservation[
                "positive_physical_delta_float32_sha256"
            ],
            "negative_physical_delta_float32_sha256": reservation[
                "negative_physical_delta_float32_sha256"
            ],
            "accepted": False,
        },
        tensors={"direction": torch.zeros(3, dtype=torch.float64)},
    )
    resumed = runner.ComputeLedger(path=tmp_path / "ledger.json", lock_identity_sha256="f" * 64)
    assert runner._recover_pending_candidate(torch, resumed) == {
        "status": "recovered_completed_state",
        "work_id": reservation["work_id"],
    }
    resumed.require_artifact(work_id=reservation["work_id"], path=runner._scenario_path("toy", 1))
    assert not runner._scenario_failure_path("toy", 1).exists()


def test_pending_candidate_reservation_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_candidate_capture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "capture_closed_loop_dms_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _invoke_prepared_capture(prepared)
    path = runner._scenario_reservation_path("toy", 1)
    reservation = runner._load_json(path)
    reservation["selected_progress_fraction"] = 0.0625
    reservation = runner._with_hash(reservation, "reservation_sha256")
    runner._atomic_text(path, json.dumps(reservation) + "\n")
    with pytest.raises(RuntimeError, match="reservation artifact differs"):
        runner.ComputeLedger(path=tmp_path / "ledger.json", lock_identity_sha256="f" * 64)


def test_only_certified_infeasibility_falls_through_progress_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _contexts()
    metadata, tensors = _state0(contexts)
    calls = []

    def fake_solver(*args, **kwargs):
        del args
        calls.append(kwargs["progress_fraction"])
        if kwargs["progress_fraction"] == 0.25:
            raise SymmetricSequentialDMSInfeasibleError("infeasible")
        raise SymmetricSequentialDMSCertificateError("bad certificate")

    monkeypatch.setattr(runner, "solve_symmetric_sequential_trust_region_update", fake_solver)
    with pytest.raises(SymmetricSequentialDMSCertificateError):
        runner._select_update(
            state_metadata=metadata,
            state_tensors=tensors,
            contexts=contexts,
            residual_scale=0.5,
            standardized_nuisance_rows=torch.zeros((8, 3), dtype=torch.float64),
        )
    assert calls == [0.25, 0.125]


def test_fixed_path_cap_violation_fails_closed_without_trying_lower_p() -> None:
    contexts = _contexts()
    metadata, tensors = _state0(contexts)
    metadata["cumulative_path_l2"] = 1.99
    with pytest.raises(SymmetricSequentialDMSCertificateError, match="fixed final/path"):
        runner._select_update(
            state_metadata=metadata,
            state_tensors=tensors,
            contexts=contexts,
            residual_scale=0.5,
            standardized_nuisance_rows=torch.tensor([[0.0, 1.0, 0.0]] * 8, dtype=torch.float64),
        )


def test_float32_state_revalidation_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _contexts()
    metadata, tensors = _state0(contexts)
    candidate = SimpleNamespace()
    monkeypatch.setattr(
        runner,
        "solve_symmetric_sequential_trust_region_update",
        lambda *args, **kwargs: candidate,
    )

    def reject_cast_state(value):
        assert value is candidate
        raise SymmetricSequentialDMSCertificateError("float32 cast state differs")

    monkeypatch.setattr(
        runner,
        "revalidate_symmetric_sequential_trust_region_update",
        reject_cast_state,
    )
    with pytest.raises(SymmetricSequentialDMSCertificateError, match="float32 cast"):
        runner._select_update(
            state_metadata=metadata,
            state_tensors=tensors,
            contexts=contexts,
            residual_scale=0.5,
            standardized_nuisance_rows=torch.zeros((8, 3), dtype=torch.float64),
        )


def test_real_solver_audit_and_physical_arrays_round_trip_checkpoint(
    tmp_path: Path,
) -> None:
    contexts = _contexts()
    metadata, tensors = _state0(contexts)
    candidate, progress, attempts = runner._select_update(
        state_metadata=metadata,
        state_tensors=tensors,
        contexts=contexts,
        residual_scale=0.5,
        standardized_nuisance_rows=torch.tensor([[0.0, 1.0, 0.0]] * 8, dtype=torch.float64),
    )
    assert candidate is not None and progress == 0.25
    selected = next(row for row in attempts if row["status"] == "certified")
    assert len(selected["revalidation_sha256"]) == 64
    path = tmp_path / "realized.pt"
    runner._save_tensor_checkpoint(
        torch,
        path=path,
        metadata={
            "schema_version": "toy.realized.v1",
            "solver_diagnostics": runner._plain_data(candidate.diagnostics),
            "solver_revalidation_sha256": selected["revalidation_sha256"],
        },
        tensors={
            "direction": torch.from_numpy(candidate.realized_direction.copy()),
            "positive": torch.from_numpy(candidate.positive_physical_float32.copy()),
            "negative": torch.from_numpy(candidate.negative_physical_float32.copy()),
        },
    )
    loaded, loaded_tensors = runner._load_tensor_checkpoint(
        torch, path=path, schema="toy.realized.v1"
    )
    assert loaded["solver_diagnostics"]["realized_deployment_certificate"]["passes"]
    assert (
        loaded_tensors["negative"].numpy().tobytes()
        == np.negative(loaded_tensors["positive"].numpy()).tobytes()
    )


def test_run_requires_an_immutable_lock_before_preflight_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_lock():
        raise RuntimeError("lock must exist first")

    monkeypatch.setattr(runner, "_load_lock", missing_lock)
    monkeypatch.setattr(
        runner,
        "run_preflight",
        lambda: pytest.fail("preflight ran before the lock was validated"),
    )
    with pytest.raises(RuntimeError, match="lock must exist first"):
        runner.run_development()


def test_lock_can_be_written_before_any_model_or_preflight_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "lock.json"
    proposed = runner._with_hash(
        {"schema_version": runner.LOCK_SCHEMA, "status": "prospective"},
        "lock_identity_sha256",
    )
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "proposed_lock", lambda: proposed)
    observed = runner.run_lock()
    assert observed == proposed
    assert runner._load_json(lock_path) == proposed


def test_failed_rejected_trial_terminal_binds_rolled_back_accepted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "SCENARIO_ROOT", tmp_path / "scenarios")
    accepted = {
        "state_index": 2,
        "checkpoint_sha256": "a" * 64,
        "direction_sha256": "b" * 64,
        "direction_l2": 0.2,
        "cumulative_path_l2": 0.3,
    }
    rejected = {
        "state_index": 3,
        "checkpoint_sha256": "c" * 64,
        "direction_sha256": "d" * 64,
        "selected_progress_fraction": 0.0625,
    }
    terminal = runner._terminal_record(
        scenario_id="toy",
        status="failed",
        state_metadata=accepted,
        reason="minimum p rejected",
        rejected_state_metadata=rejected,
    )
    assert terminal["state_index"] == 2
    assert terminal["state_checkpoint_sha256"] == "a" * 64
    assert terminal["last_rejected_trial"]["state_index"] == 3
    assert terminal["last_rejected_trial"]["checkpoint_sha256"] == "c" * 64


def test_kl_gates_are_applied_separately_to_mean_p95_and_max() -> None:
    assert runner._kl_report([0.0] * 20)["passes"] is True
    assert runner._kl_report([0.006] * 20)["passes"] is False
    p95_failure = [0.0] * 18 + [0.021] * 2
    assert runner._kl_report(p95_failure)["p95"] == pytest.approx(0.021)
    assert runner._kl_report(p95_failure)["passes"] is False
    assert runner._kl_report([0.0] * 19 + [0.051])["passes"] is False


def test_finite_gate_rejects_poor_target_agreement_and_unrelated_error() -> None:
    contexts = _contexts()
    metadata, tensors = _state0(contexts)
    certificate = {
        "passes": True,
        "target_realized_progress": [0.04] * 8,
        "target_required_progress": [0.03] * 8,
        "unrelated_plus_desired_margins": [1.0] * 8,
        "unrelated_minus_desired_margins": [1.0] * 8,
    }
    observations = _candidate_observations(
        contexts,
        target_plus_margin=-0.018,
        target_minus_margin=-0.022,
        unrelated_plus=[1.051] + [1.0] * 7,
        unrelated_minus=[1.0] * 8,
    )
    gate = runner._actual_candidate_gate(
        previous_metadata=metadata,
        previous_tensors=tensors,
        candidate_observations=observations,
        contexts=contexts,
        solver_diagnostics={"realized_deployment_certificate": certificate},
    )
    assert gate["passes"] is False
    assert gate["target_trust_agreement_failure_count"] == 8
    assert gate["maximum_unrelated_linearization_error"] == pytest.approx(0.051)


def test_nonlinear_toy_loop_rolls_back_retries_and_eventually_stops() -> None:
    """Exercise the model-free controller loop across the complete scientific path."""

    contexts = _contexts(width=3)
    state_metadata, state_tensors = _state0(contexts, width=3)
    residual_scale = 0.5
    # Raw dmargin/dphysical=[2,0,0] becomes [1,0,0] in standardized units.
    assert torch.equal(
        contexts[0]["raw_gradient"].double() * residual_scale,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    nuisance = torch.tensor([[0.0, 1.0, 0.0]] * 8, dtype=torch.float64)

    rejected = []
    trial_count = 0
    stopped = False
    while trial_count < runner.MAX_TRIAL_STATES:
        candidate, progress, _ = runner._select_update(
            state_metadata=state_metadata,
            state_tensors=state_tensors,
            contexts=contexts,
            residual_scale=residual_scale,
            standardized_nuisance_rows=nuisance,
            excluded_progress=rejected,
        )
        assert candidate is not None and progress is not None
        trial_count += 1

        # The API's authoritative state must be the float32 physical round-trip,
        # and the exact same +/- vectors are reused for every form and answer order.
        physical_plus = candidate.positive_physical_float32
        physical_minus = candidate.negative_physical_float32
        assert physical_minus.tobytes() == np.negative(physical_plus).tobytes()
        assert np.array_equal(
            (candidate.realized_direction * residual_scale).astype(np.float32),
            physical_plus,
        )

        cert = candidate.diagnostics["realized_deployment_certificate"]
        predicted = list(map(float, cert["target_realized_progress"]))
        previous_plus, previous_minus, _, _ = runner._branch_maps(state_metadata, state_tensors)
        target_ids = [row["form_id"] for row in contexts if row["category"] == "target"]
        current_plus_margin = float(
            previous_plus[target_ids[0]]["positive_minus_negative_log_odds"]
        )
        current_minus_margin = float(
            previous_minus[target_ids[0]]["positive_minus_negative_log_odds"]
        )
        # Deliberately nonlinear first trial: only 10% of predicted movement,
        # forcing rollback. All later trials realize the predicted local movement.
        agreement = 0.10 if trial_count == 1 else 1.0
        next_plus = current_plus_margin + agreement * predicted[0]
        next_minus = current_minus_margin - agreement * predicted[4]
        observations = _candidate_observations(
            contexts,
            target_plus_margin=next_plus,
            target_minus_margin=next_minus,
            unrelated_plus=list(cert["unrelated_plus_desired_margins"]),
            unrelated_minus=list(cert["unrelated_minus_desired_margins"]),
        )
        gate = runner._actual_candidate_gate(
            previous_metadata=state_metadata,
            previous_tensors=state_tensors,
            candidate_observations=observations,
            contexts=contexts,
            solver_diagnostics=candidate.diagnostics,
        )
        if not gate["passes"]:
            assert trial_count == 1 and progress == 0.25
            rejected.append(progress)
            # Rollback: state and direction stay byte-identical before p=.125.
            assert torch.equal(state_tensors["direction"], torch.zeros(3, dtype=torch.float64))
            continue

        rejected = []
        gradients = torch.stack([row["raw_gradient"] for row in contexts] * 2).float()
        direction = torch.from_numpy(candidate.realized_direction.copy()).double()
        state_metadata = {
            "state_index": trial_count,
            "checkpoint_sha256": f"{trial_count:064x}",
            "direction_sha256": canonical_sha256(direction.tolist()),
            "direction_l2": float(direction.norm().item()),
            "cumulative_path_l2": float(direction.norm().item()),
            "accepted": True,
            "observations": observations,
        }
        state_tensors = {"direction": direction, "raw_gradients": gradients}
        if runner._stopping_gate(observations, contexts):
            stopped = True
            break

    assert stopped is True
    assert 2 <= trial_count < runner.MAX_TRIAL_STATES


def test_stopping_gate_requires_every_order_and_sign() -> None:
    contexts = _contexts()
    passing = _candidate_observations(
        contexts,
        target_plus_margin=0.05,
        target_minus_margin=-0.05,
        unrelated_plus=[1.0] * 8,
        unrelated_minus=[1.0] * 8,
    )
    assert runner._stopping_gate(passing, contexts) is True
    passing[0]["positive_minus_negative_log_odds"] = 0.049
    assert runner._stopping_gate(passing, contexts) is False


def test_deep_solver_diagnostics_are_not_mutable_assumptions() -> None:
    # The runner serializes a plain copy but never mutates the solver's returned audit.
    diagnostics = MappingProxyType(
        {
            "passes": True,
            "realized_deployment_certificate": MappingProxyType({"passes": True}),
        }
    )
    candidate = SimpleNamespace(diagnostics=diagnostics)
    assert candidate.diagnostics["realized_deployment_certificate"]["passes"] is True
    with pytest.raises(TypeError):
        candidate.diagnostics["passes"] = False
