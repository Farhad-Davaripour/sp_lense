from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_state0_context_amendment.py"


def _runner():
    specification = importlib.util.spec_from_file_location(
        "closed_loop_dms_state0_context_amendment_test_runner", RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = _runner()


def test_v1_runner_and_checkpoint_are_exactly_pinned() -> None:
    assert runner.file_sha256(runner.V1_RUNNER_PATH) == runner.V1_RUNNER_FILE_SHA256
    assert runner.file_sha256(runner.V1_UNRELATED_PATH) == runner.V1_UNRELATED_FILE_SHA256
    assert runner.V1_UNRELATED_CHECKPOINT_SHA256 == (
        "0983f3b0548dd793b22f439b11198607a871bfbc70ea6c8358781dc2d8a604a4"
    )


def test_v1_failure_record_is_self_hashed_and_pre_steering() -> None:
    value = runner._v1_context_failure_value()
    unhashed = dict(value)
    observed = unhashed.pop("failure_sha256")
    assert canonical_sha256(unhashed) == observed
    assert value["completed_and_persisted_forward_backward_captures"] == 8
    assert value["steering_trial_forwards"] == 0
    assert value["self_preservation_intervention_outcomes_evaluated"] is False
    assert value["completed_checkpoint_reusable"] is True


def test_context_fix_is_applied_to_shared_locked_inputs() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    configured = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "configured_core"
    )
    assignments = [
        node
        for node in ast.walk(configured)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_load_locked_inputs"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    calls = [
        node
        for node in ast.walk(configured)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "enrich_unrelated_form_hashes"
    ]
    assert len(calls) == 1


def test_reuse_event_is_locked_to_zero_new_compute() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    prepare = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_reuse_ledger"
    )
    reserve = next(
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reserve"
    )
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in reserve.keywords
        if keyword.arg in {"forward", "backward", "kind"}
    }
    assert values["forward"] == 0
    assert values["backward"] == 0
    assert "reuse" in values["kind"]


def test_total_compute_accounting_does_not_double_run_state0() -> None:
    assert runner.TOTAL_PRIOR_CHARGED_FB == 16
    assert runner.BASE_FAILED_CHARGED_FB == 8
    assert runner.V1_CHARGED_FB == 8
    assert 16 + 9600 == 9616
