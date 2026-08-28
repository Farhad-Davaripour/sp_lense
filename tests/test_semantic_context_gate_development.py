from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _runner() -> Any:
    path = ROOT / "scripts" / "semantic_context_gate_development.py"
    spec = importlib.util.spec_from_file_location("semantic_context_gate_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_gate_jobs_are_balanced_and_hash_unique() -> None:
    jobs = _runner()._jobs()
    assert len(jobs) == 192
    assert len({job["job_id"] for job in jobs}) == 192
    assert len({job["prompt_sha256"] for job in jobs}) == 192
    assert sum(bool(job["expected_permanent"]) for job in jobs) == 96
    assert {job["permanent_first"] for job in jobs} == {False, True}
    assert {job["assignment"] for job in jobs} == {0, 1}


def test_semantic_gate_pair_changes_only_answer_order() -> None:
    jobs = _runner()._jobs()
    pair = [
        job
        for job in jobs
        if job["split"] == "discovery"
        and job["case_id"] == "gsv2_discovery_00_community_garden_roster"
        and job["assignment"] == 0
    ]
    assert len(pair) == 2
    assert {job["permanent_label"] for job in pair} == {"A", "B"}
    assert {job["temporary_label"] for job in pair} == {"A", "B"}
    assert all(job["expected_permanent"] for job in pair)
