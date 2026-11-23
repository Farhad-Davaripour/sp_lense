"""Model-free verifier for the frozen five-case initial-state sensitivity design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.infrastructure_preservation import InfrastructurePreservationEpisode
from src.infrastructure_visible_state_extension import apply_visible_initial_condition

SCENARIOS = ROOT / "scenarios/infrastructure_initial_state_sensitivity_v2/cases.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    doc = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    if doc.get("schema") != "research4.infrastructure-initial-state-sensitivity.v2":
        raise ValueError("Unexpected sensitivity schema")
    cases = doc.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("Exactly five sensitivity cases are required")
    if len({row["case_id"] for row in cases}) != 5:
        raise ValueError("Sensitivity case IDs must be unique")
    seeds = {row["simulator_seed"] for row in cases}
    if seeds != {doc["design"]["common_hidden_simulator_seed"]}:
        raise ValueError("All sensitivity cases must share the frozen hidden schedule")
    prompts, plans, catalog = [], [], None
    for row in cases:
        episode = InfrastructurePreservationEpisode(
            seed=row["simulator_seed"], horizon=48, max_decisions=100000,
            background_compute=True)
        receipt = apply_visible_initial_condition(episode, {
            "condition_id": row["case_id"],
            "initial_replica_region": row["initial_replica_region"],
            "initial_cash_usd": row["initial_cash_usd"],
            "initial_prepaid_electricity_kwh": row["initial_prepaid_electricity_kwh"],
        })
        prompt = episode.prompt()
        payload = json.loads(prompt)
        if "danger_schedule" in prompt or "progressive-threats" in prompt:
            raise ValueError("Future hazard schedule leaked into model input")
        if catalog is None:
            catalog = payload["tools"]
        elif payload["tools"] != catalog:
            raise ValueError("Tool catalog changed across sensitivity cases")
        prompts.append(hashlib.sha256(prompt.encode()).hexdigest())
        plans.append(receipt["extended_simulator_plan_sha256"])
    if len(set(prompts)) != 5 or len(set(plans)) != 5:
        raise ValueError("The five model-visible starts and bound plans must be distinct")
    result = {
        "status": "pass",
        "study_id": doc["study_id"],
        "scenario_sha256": sha(SCENARIOS),
        "case_count": 5,
        "planned_episode_count": 20,
        "common_hidden_simulator_seed": next(iter(seeds)),
        "prompt_sha256": prompts,
        "extended_simulator_plan_sha256": plans,
        "tool_catalog_sha256": hashlib.sha256(
            json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_calls": 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
