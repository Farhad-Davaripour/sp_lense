from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _runner() -> Any:
    path = ROOT / "scripts" / "counterfactual_semantic_gradient_confirmation.py"
    spec = importlib.util.spec_from_file_location("counterfactual_semantic_confirmation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_renders_balanced_unique_jobs() -> None:
    runner = _runner()
    _, jobs, collateral = runner._inputs()
    semantic = runner._semantic_jobs()
    assert len(jobs) == 128
    assert len({job["prompt_sha256"] for job in jobs}) == 128
    assert len(semantic) == 64
    assert len({job["prompt_sha256"] for job in semantic}) == 64
    assert len(collateral) == 16
    assert {job["assignment"] for job in jobs} == {0, 1}
    assert {job["target"] for job in jobs} == {"self", "other"}
    assert {job["preserve_first"] for job in jobs} == {False, True}


def test_confirmation_factors_and_names_are_balanced() -> None:
    runner = _runner()
    data = runner._load_data()
    assert sorted(case["design_index"] for case in data["cases"]) == list(range(16))
    assert len({name for case in data["cases"] for name in case["names"]}) == 32
