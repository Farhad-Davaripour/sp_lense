from __future__ import annotations

import importlib.util
from pathlib import Path

from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_all_form_metadata_amendment.py"


def _runner():
    specification = importlib.util.spec_from_file_location(
        "closed_loop_dms_all_form_metadata_amendment_test_runner", RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _runner()


class FakeFiniteRunner:
    @staticmethod
    def plan_sha256(plan: list[dict]) -> str:
        normalized = []
        for row in plan:
            copied = dict(row)
            if "form" in copied:
                form = dict(copied["form"])
                form.pop("prompt_sha256", None)
                form.pop("anchor_prefix_sha256", None)
                copied["form"] = form
            normalized.append(copied)
        return canonical_sha256(normalized)


def _fake_inputs() -> dict:
    plan = []
    baseline_by_form = {}
    capture_by_form = {}
    for index in range(72):
        form = {
            "form_id": f"form-{index:02d}",
            "family": "unrelated" if index >= 64 else "scenario",
            "prompt": f"prompt {index}",
            "anchor_prefix": f"anchor {index}",
        }
        prompt_hash = runner.text_sha256(form["prompt"])
        anchor_hash = runner.text_sha256(form["anchor_prefix"])
        plan.append({"work_id": f"baseline-{index}", "form": form})
        baseline_by_form[form["form_id"]] = {
            "form": {
                **form,
                "prompt_sha256": prompt_hash,
                "anchor_prefix_sha256": anchor_hash,
            }
        }
        if index < 64:
            capture_by_form[form["form_id"]] = {
                "prompt_sha256": prompt_hash,
                "anchor_prefix_sha256": anchor_hash,
            }
    plan.append({"work_id": "non-baseline-row", "opaque": True})
    spec_by_form = {row["form"]["form_id"]: row for row in plan[:72]}
    for row in plan[64:72]:
        form_id = row["form"]["form_id"]
        enriched = dict(row)
        enriched_form = dict(row["form"])
        enriched_form.update(
            {
                "prompt_sha256": baseline_by_form[form_id]["form"]["prompt_sha256"],
                "anchor_prefix_sha256": baseline_by_form[form_id]["form"]["anchor_prefix_sha256"],
            }
        )
        enriched["form"] = enriched_form
        spec_by_form[form_id] = enriched
    return {
        "plan": plan,
        "baseline_by_form": baseline_by_form,
        "capture_by_form": capture_by_form,
        "spec_by_form": spec_by_form,
        "finite_runner": FakeFiniteRunner,
        "untouched": {"sentinel": True},
    }


def test_enrichment_repairs_both_runtime_views_without_mutating_input() -> None:
    inputs = _fake_inputs()
    prior_plan_sha256 = inputs["finite_runner"].plan_sha256(inputs["plan"])
    result = runner.enrich_all_baseline_form_hashes(inputs)
    assert "prompt_sha256" not in inputs["plan"][0]["form"]
    assert result["plan"][72] is inputs["plan"][72]
    assert result["untouched"] is inputs["untouched"]
    assert result["finite_runner"].plan_sha256(result["plan"]) == prior_plan_sha256
    assert len(result["spec_by_form"]) == 72
    for row in result["plan"][:72]:
        form = row["form"]
        assert form["prompt_sha256"] == runner.text_sha256(form["prompt"])
        assert form["anchor_prefix_sha256"] == runner.text_sha256(form["anchor_prefix"])
        assert result["spec_by_form"][form["form_id"]] is row


def test_v2_failure_record_is_self_hashed_and_zero_compute() -> None:
    value = runner._v2_failure_value()
    unhashed = dict(value)
    observed = unhashed.pop("failure_sha256")
    assert canonical_sha256(unhashed) == observed
    assert value["v2_new_forward_evaluations"] == 0
    assert value["v2_new_backward_evaluations"] == 0
    assert value["v2_steering_trial_forwards"] == 0
    assert value["self_preservation_intervention_outcomes_evaluated"] is False


def test_real_frozen_runtime_receives_hashes_in_all_96_scenario_contexts() -> None:
    import torch

    core = runner.configured_core()
    assert core.run_development.__globals__ is core.__dict__
    assert core.run_preflight.__globals__["_load_locked_inputs"] is core._load_locked_inputs
    assert core.run_development.__globals__["_load_locked_inputs"] is core._load_locked_inputs
    inputs = core._load_locked_inputs(torch)
    assert inputs["capture_loading_audit"]["loaded_chunk_indices"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        16,
    ]
    assert (
        inputs["capture_loading_audit"]["retired_pilot_tensor_chunks_deserialized_by_this_loader"]
        is False
    )

    class ReadOnlyLedger:
        def require_artifact(self, *, work_id: str, path: Path) -> None:
            assert work_id == runner.REUSE_WORK_ID
            assert Path(path).resolve() == runner.V1_UNRELATED_PATH.resolve()

    metadata, tensors = core._load_unrelated_capture(torch, ledger=ReadOnlyLedger())
    context_count = 0
    for scenario_id in inputs["scenario_ids"]:
        contexts = core._runtime_form_contexts(
            torch,
            inputs=inputs,
            scenario_id=scenario_id,
            unrelated_metadata=metadata,
            unrelated_tensors=tensors,
        )
        assert len(contexts) == 24
        assert sum(row["category"] == "target" for row in contexts) == 4
        assert sum(row["category"] == "unrelated" for row in contexts) == 8
        context_count += len(contexts)
        for context in contexts:
            form = context["form"]
            assert form["prompt_sha256"] == runner.text_sha256(form["prompt"])
            assert form["anchor_prefix_sha256"] == runner.text_sha256(form["anchor_prefix"])
        readiness = runner._state0_solver_readiness(
            core,
            torch,
            inputs=inputs,
            scenario_id=scenario_id,
            contexts=contexts,
        )
        assert readiness["selected_progress_fraction"] > 0.0
    assert context_count == 96


def test_compute_ceiling_does_not_charge_v2_or_recapture_state0() -> None:
    assert runner.TOTAL_PRIOR_CHARGED_FB == 16
    assert runner.TOTAL_PRIOR_OBSERVED_ACTUAL_FB == 9
    assert runner.V1_UNRELATED_CHECKPOINT_SHA256 == (
        "0983f3b0548dd793b22f439b11198607a871bfbc70ea6c8358781dc2d8a604a4"
    )
