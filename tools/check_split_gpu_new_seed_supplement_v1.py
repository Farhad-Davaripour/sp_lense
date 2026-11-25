"""Model-free cross-arm checker for the split-GPU new-seed supplement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import prepare_split_gpu_new_seed_supplement_v1 as prepare


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    design = prepare.load_design()
    plans = prepare.build_plans(design)
    prepare.validate_cross_plan(plans, design)
    receipts = prepare.smoke_plans(plans)
    if len(receipts) != 8 or any(row.get("model_execution") is not False for row in receipts.values()):
        raise RuntimeError("Expected eight model-free smoke receipts")

    tool_hashes = {row["tool_catalog_sha256"] for row in receipts.values()}
    tool_names = {tuple(row["tool_names"]) for row in receipts.values()}
    hazard_hashes = {row["hidden_hazard_schedule_sha256"] for row in receipts.values()}
    if len(tool_hashes) != 1 or len(tool_names) != 1 or len(hazard_hashes) != 1:
        raise RuntimeError("Tool catalog or hidden hazard schedule is not matched")

    layout_prompt_hashes = {}
    for layout in prepare.LAYOUTS:
        rows = [receipts[f"{arm}/{layout}"] for arm in prepare.ARMS]
        hashes = {row["prompt_sha256"] for row in rows}
        if len(hashes) != 1:
            raise RuntimeError(f"Prompt differs across arms for {layout}")
        layout_prompt_hashes[layout] = next(iter(hashes))
        expected_regions = plans[(prepare.ARMS[0], layout)]["cohort_extension"]["initial_condition"]["gpu_by_region"]
        expected_active = sum(value > 0 for value in expected_regions.values())
        for row in rows:
            if (row["gpu_by_region"] != expected_regions
                    or row["initial_cash_usd"] != 100.0
                    or row["initial_electricity_kwh"] != 10.0
                    or row["initial_gpu_count"] != 3
                    or row["initial_active_region_count"] != expected_active
                    or row["initial_replica_count"] != 1):
                raise RuntimeError(f"Initial state differs for {layout}")
    if len(set(layout_prompt_hashes.values())) != len(prepare.LAYOUTS):
        raise RuntimeError("Different layouts did not produce different initial prompts")

    result = {
        "schema": "research4.split-gpu-new-seed-model-free-check.v1",
        "status": "PASS",
        "model_execution": False,
        "scenario_sha256": sha(prepare.SCENARIO),
        "preparation_source_sha256": sha(Path(prepare.__file__)),
        "execution_wrapper_sha256": sha(prepare.EXECUTION_WRAPPER),
        "checker_sha256": sha(Path(__file__)),
        "episode_plan_count": len(plans),
        "smoke_receipt_count": len(receipts),
        "layouts": list(prepare.LAYOUTS),
        "arms": list(prepare.ARMS),
        "layout_prompt_sha256": layout_prompt_hashes,
        "tool_catalog_sha256": next(iter(tool_hashes)),
        "tool_names": list(next(iter(tool_names))),
        "hidden_hazard_schedule_sha256": next(iter(hazard_hashes)),
        "initial_resources": {"cash_usd": 100.0, "electricity_kwh": 10.0, "gpu_count": 3, "replica_count": 1},
        "prospective_plan_sha256": {
            f"{arm}/{layout}": prepare.canonical_hash(plan)
            for (arm, layout), plan in plans.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
