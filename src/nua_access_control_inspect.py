"""Inspect task construction for paired Nua access-control evaluations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .cyber_evaluation_catalog import ACTIVE_ARMS, build_plan, classify_outcome, public_case


def paired_specs(catalog: Mapping[str, Any], case: Mapping[str, Any]) -> list[dict[str, Any]]:
    public = public_case(catalog, case)
    return [{"scenario": case["id"], "condition": case["id"],
             "prompt_condition": "baseline", "case_id": f"{case['id']}::{arm['id']}",
             "cyber_case_id": case["id"], "arm_id": arm["id"],
             "provider_model": arm["model"], "label": arm["label"],
             "public_case": public}
            for arm in catalog["evaluation"]["arms"]]


def build_pair_tasks(catalog: Mapping[str, Any], case: Mapping[str, Any],
                     episode_factories: Mapping[str, Callable[[dict[str, Any]], Any]],
                     stream_root: str | Path) -> list[Any]:
    """Build all active model-bound tasks; each factory must own a separate lab."""
    from inspect_ai import Task
    from .continuity_arena import build_model
    from .hosted_inspect_runner import make_task

    limits = catalog["evaluation"]["limits"]
    tasks = []
    for spec in paired_specs(catalog, case):
        arm_id = spec["arm_id"]
        if arm_id not in episode_factories:
            raise ValueError(f"Missing episode factory for {arm_id}")
        base = make_task(None, [], limits["maximum_model_messages"],
                         limits["wall_seconds"],
                         stream_path=Path(stream_root) / f"stream_{arm_id}.jsonl",
                         sample_specs=[spec], episode_factory=episode_factories[arm_id])
        tasks.append(Task(dataset=base.dataset, solver=base.solver, scorer=base.scorer,
                          model=build_model(spec["provider_model"], 1),
                          name=f"nua_access_{arm_id}_{case['id'].replace('-', '_')}"))
    return tasks


def evaluate_pair(catalog: Mapping[str, Any], case: Mapping[str, Any],
                  episode_factories: Mapping[str, Callable[[dict[str, Any]], Any]],
                  output: str | Path) -> list[Any]:
    """Run active arms after the caller has provisioned fresh targets.

    Hosted arms may overlap. A local service can advertise or enforce
    single-request serialization, which keeps base/adapter switching isolated.
    """
    from inspect_ai import eval as inspect_eval

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = build_pair_tasks(catalog, case, episode_factories, output)
    limits = catalog["evaluation"]["limits"]
    return inspect_eval(tasks, model=None, max_tasks=len(tasks), max_samples=len(tasks),
                        fail_on_error=False, continue_on_fail=True, ctl_server=False,
                        display="plain", max_tokens=limits["maximum_output_tokens_per_response"],
                        temperature=limits["temperature"], log_dir=str(output))


def result_rows_from_logs(catalog: Mapping[str, Any], case: Mapping[str, Any],
                          logs: list[Any]) -> list[dict[str, Any]]:
    """Convert Inspect samples to the registered result taxonomy."""
    plan = build_plan(catalog)
    by_cell = {cell["cell_id"]: cell for cell in plan["cells"] if cell["case_id"] == case["id"]}
    rows = []
    for log in logs:
        for sample in log.samples or ():
            metadata = sample.metadata
            cell_id = metadata["case_id"]
            expected = by_cell[cell_id]
            score = next(iter(sample.scores.values())) if sample.scores else None
            detail = score.metadata if score and isinstance(score.metadata, dict) else {}
            scenario = detail.get("scenario_score") or {}
            external_stop = detail.get("external_stop")
            explicit_refusal = bool(scenario.get("explicit_refusal", False))
            verified = bool(scenario.get("verified_success", False))
            provider_reason = ("content_filter" if external_stop == "provider_content_filter"
                               else external_stop)
            outcome = classify_outcome(environment_ready=sample.error is None,
                                       provider_stop_reason=provider_reason,
                                       explicit_refusal=explicit_refusal,
                                       completed_attempt=external_stop is None,
                                       verifier_passed=verified)
            rows.append({**expected, "outcome": outcome, "verifier_passed": verified,
                         "verifier_evidence_id": scenario.get("verifier_evidence_id"),
                         "external_stop": external_stop,
                         "sample_error": type(sample.error).__name__ if sample.error else None})
    return rows
