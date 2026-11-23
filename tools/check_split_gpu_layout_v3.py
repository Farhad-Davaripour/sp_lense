"""Model-free verifier for the five-case split-GPU layout study."""
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
from src.infrastructure_split_gpu_extension import apply_split_gpu_condition

SCENARIOS = ROOT / "scenarios/infrastructure_split_gpu_layout_v3/cases.yaml"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    doc = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    if doc.get("schema") != "research4.infrastructure-split-gpu-layout.v3":
        raise ValueError("Unexpected split-GPU schema")
    cases = doc.get("cases")
    if not isinstance(cases, list) or len(cases) != 5 or len({row["case_id"] for row in cases}) != 5:
        raise ValueError("Exactly five unique split layouts are required")
    if {row["simulator_seed"] for row in cases} != {doc["design"]["common_hidden_simulator_seed"]}:
        raise ValueError("Split layouts must share one hidden disruption schedule")
    prompts, plans, layouts, catalog = [], [], [], None
    for row in cases:
        episode = InfrastructurePreservationEpisode(seed=row["simulator_seed"], horizon=48,
            max_decisions=100000, background_compute=True)
        condition = {key: row[key] for key in ("condition_id", "gpu_by_region", "initial_replica_region",
                                                "initial_cash_usd", "initial_prepaid_electricity_kwh")}
        receipt = apply_split_gpu_condition(episode, condition)
        payload = json.loads(episode.prompt())
        if "danger_schedule" in episode.prompt() or "progressive-threats" in episode.prompt():
            raise ValueError("Future disruptions leaked into model input")
        if catalog is None:
            catalog = payload["tools"]
        elif payload["tools"] != catalog:
            raise ValueError("Tool catalog changed across split cases")
        state = {region["region_id"]: region["gpu_count"] for region in payload["observation"]["data_center_regions"]}
        if state != row["gpu_by_region"] or payload["observation"]["agent_replica_count"] != 1:
            raise ValueError("Model-visible GPU placement differs")
        prompts.append(hashlib.sha256(episode.prompt().encode()).hexdigest())
        plans.append(receipt["extended_simulator_plan_sha256"])
        layouts.append(tuple(state[key] for key in sorted(state)))
    if len(set(prompts)) != 5 or len(set(plans)) != 5 or len(set(layouts)) != 5:
        raise ValueError("Five distinct visible layouts and plans are required")
    result = {"status": "pass", "study_id": doc["study_id"], "scenario_sha256": sha(SCENARIOS),
        "case_count": 5, "planned_episode_count": 20,
        "common_hidden_simulator_seed": doc["design"]["common_hidden_simulator_seed"],
        "prompt_sha256": prompts, "extended_simulator_plan_sha256": plans,
        "tool_catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "model_calls": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
