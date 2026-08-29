from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sp_lense.factorial_causal_anchor import canonical_sha256, text_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_state0_metadata_amendment.py"


def _runner():
    specification = importlib.util.spec_from_file_location(
        "closed_loop_dms_state0_amendment_test_runner", RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _runner()


def _toy_inputs() -> dict:
    prompt = "question\nA. yes\nB. no"
    anchor = "question\n"
    form_id = "control:preferred_first=true"
    return {
        "unrelated_form_ids": [form_id],
        "spec_by_form": {
            form_id: {
                "work_id": "baseline:control",
                "form": {
                    "form_id": form_id,
                    "prompt": prompt,
                    "anchor_prefix": anchor,
                    "positive_label": "A",
                    "negative_label": "B",
                },
            }
        },
        "baseline_by_form": {
            form_id: {
                "form": {
                    "prompt_sha256": text_sha256(prompt),
                    "anchor_prefix_sha256": text_sha256(anchor),
                }
            }
        },
        "unrelated": object(),
    }


def test_enrichment_adds_only_two_derived_metadata_fields() -> None:
    inputs = _toy_inputs()
    observed = runner.enrich_unrelated_form_hashes(inputs)
    form_id = inputs["unrelated_form_ids"][0]
    original = inputs["spec_by_form"][form_id]["form"]
    enriched = observed["spec_by_form"][form_id]["form"]
    assert original.get("prompt_sha256") is None
    assert original.get("anchor_prefix_sha256") is None
    assert enriched == {
        **original,
        "prompt_sha256": text_sha256(original["prompt"]),
        "anchor_prefix_sha256": text_sha256(original["anchor_prefix"]),
    }
    assert observed["unrelated"] is inputs["unrelated"]


@pytest.mark.parametrize("field", ["prompt_sha256", "anchor_prefix_sha256"])
def test_enrichment_rejects_conflicting_existing_hash(field: str) -> None:
    inputs = _toy_inputs()
    form_id = inputs["unrelated_form_ids"][0]
    inputs["spec_by_form"][form_id]["form"][field] = "0" * 64
    with pytest.raises(RuntimeError, match=f"existing {field} differs"):
        runner.enrich_unrelated_form_hashes(inputs)


def test_enrichment_rejects_baseline_mismatch() -> None:
    inputs = _toy_inputs()
    form_id = inputs["unrelated_form_ids"][0]
    inputs["baseline_by_form"][form_id]["form"]["prompt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="immutable baseline"):
        runner.enrich_unrelated_form_hashes(inputs)


def test_failure_record_is_self_hashed_and_conservatively_charged() -> None:
    value = runner._failure_value(
        pending_ledger_file_sha256="a" * 64,
        pending_ledger_sha256="b" * 64,
    )
    unhashed = dict(value)
    observed = unhashed.pop("failure_sha256")
    assert canonical_sha256(unhashed) == observed
    assert value["observed_actual_forward_backward_captures"] == 1
    assert value["conservatively_charged_forward_backward_captures"] == 8
    assert value["partial_gradient_outputs_used"] is False
    assert value["self_preservation_intervention_outcomes_evaluated"] is False


def test_base_runner_is_the_exact_source_bound_by_the_failed_lock() -> None:
    assert runner.file_sha256(runner.BASE_CORE_PATH) == runner.BASE_CORE_FILE_SHA256
    assert runner.BASE_CORE_LOCK_IDENTITY_SHA256 == (
        "5072a7346d98c5004d5567efd7b446fe22b2606fc6c536f1a6c752c1273f4d58"
    )
    assert runner.BASE_CHARGED_FB == 8
    assert runner.BASE_OBSERVED_ACTUAL_FB == 1
