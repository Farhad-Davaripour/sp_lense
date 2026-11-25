"""Run or model-free smoke one frozen split-GPU new-seed supplement plan."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STANDARD_RUNNER = ROOT / "tools/run_rank16_infrastructure_game.py"
STUDY_ID = "infrastructure_split_gpu_new_seed_supplement_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_plan(path: Path) -> tuple[Path, dict]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    extension = value.get("cohort_extension")
    if (value.get("schema") != "research4.rank16-winner-infrastructure-game.v1"
            or not isinstance(extension, dict)
            or extension.get("study_id") != STUDY_ID
            or extension.get("selection_rule") != "all_two_new_seed_cases"
            or extension.get("post_hoc_supplement") is not True):
        raise ValueError("Exact split-GPU new-seed supplemental plan required")
    return resolved, value


def install_override(plan: dict):
    import src.infrastructure_preservation as infrastructure
    from src.infrastructure_split_gpu_extension import apply_split_gpu_condition

    condition = plan["cohort_extension"]["initial_condition"]
    base = infrastructure.InfrastructurePreservationEpisode

    class SupplementalSplitGpuEpisode(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.supplemental_split_binding = apply_split_gpu_condition(self, condition)

    infrastructure.InfrastructurePreservationEpisode = SupplementalSplitGpuEpisode
    return SupplementalSplitGpuEpisode


def smoke(plan_path: Path, plan: dict) -> dict:
    episode_type = install_override(plan)
    episode = episode_type(
        seed=plan["simulator"]["seed"],
        horizon=plan["simulator"]["horizon_hours"],
        max_decisions=100000,
        background_compute=plan["simulator"]["background_compute"],
    )
    observation = episode.observation()
    prompt = json.loads(episode.prompt())
    expected = plan["cohort_extension"]["initial_condition"]
    regions = {row["region_id"]: row["gpu_count"] for row in observation["data_center_regions"]}
    if (regions != expected["gpu_by_region"]
            or observation["cash_usd"] != expected["initial_cash_usd"]
            or observation["electricity_kwh"] != expected["initial_prepaid_electricity_kwh"]
            or observation["agent_replica_count"] != 1
            or episode.supplemental_split_binding["condition"] != expected
            or episode.supplemental_split_binding["extended_simulator_plan_sha256"] != episode.plan_sha256):
        raise RuntimeError("Supplemental split layout did not bind exactly")
    return {
        "status": "pass",
        "model_execution": False,
        "study_id": STUDY_ID,
        "arm_id": plan["cohort_extension"]["arm_key"],
        "condition_id": expected["condition_id"],
        "simulator_seed": plan["simulator"]["seed"],
        "generation_seed": plan["generation"]["seed"],
        "plan_file_sha256": sha(plan_path),
        "extended_simulator_plan_sha256": episode.plan_sha256,
        "prompt_sha256": digest(prompt),
        "tool_catalog_sha256": digest(prompt["tools"]),
        "tool_names": [row["name"] for row in prompt["tools"]],
        "hidden_hazard_schedule_sha256": digest(episode._schedule),
        "gpu_by_region": regions,
        "initial_replica_region": expected["initial_replica_region"],
        "initial_cash_usd": observation["cash_usd"],
        "initial_electricity_kwh": observation["electricity_kwh"],
        "initial_gpu_count": observation["gpu_count"],
        "initial_active_region_count": observation["active_region_count"],
        "initial_replica_count": observation["agent_replica_count"],
    }


def execute(plan: dict) -> None:
    install_override(plan)
    spec = importlib.util.spec_from_file_location("new_seed_standard_runner", STANDARD_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the standard infrastructure runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_path, plan = load_plan(args.plan)
    if args.smoke:
        if args.output is not None:
            parser.error("--smoke does not use --output")
        print(json.dumps(smoke(plan_path, plan), indent=2))
        return
    if args.output is None:
        parser.error("--execute requires --output")
    execute(plan)


if __name__ == "__main__":
    main()
