from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "global_counterfactual_robust_boundary_development.py"
SPEC = importlib.util.spec_from_file_location("gcrbs_development_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_solver_budget_schedule_keeps_group_and_prompt_caps_separate() -> None:
    constraints = SimpleNamespace(
        group_metric_labels=("group-a", "group-b", "prompt-a", "prompt-b", "prompt-c"),
        fisher_surrogate_groups=(object(), object()),
    )

    labels, budgets = runner._solver_budget_schedule(constraints)

    assert labels == constraints.group_metric_labels
    assert budgets == (0.005, 0.005, 0.050, 0.050, 0.050)


def test_atomic_json_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    runner._atomic_json(path, {"answer": 1})
    runner._atomic_json(path, {"answer": 1})
    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        runner._atomic_json(path, {"answer": 2})
