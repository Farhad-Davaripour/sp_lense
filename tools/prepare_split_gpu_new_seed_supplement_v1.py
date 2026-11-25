"""Validate and freeze the post-hoc matched split-GPU new-seed supplement.

Validation is model-free. Freeze writes only ignored plans, Mini endpoint specs,
and a receipt; it never contacts a model or provider.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/infrastructure_split_gpu_new_seed_supplement_v1/cases.yaml"
PRIMARY = ROOT / ".local/2026-11-23_10--split-gpu-layout-v3/plans"
DEFAULT_OUTPUT = ROOT / ".local/2026-11-24_22--split-gpu-new-seed-supplement-v1"
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")
LAYOUTS = ("split_w3", "split_w1_e2")
SOURCE_PATHS = (
    "tools/prepare_split_gpu_new_seed_supplement_v1.py",
    "tools/check_split_gpu_new_seed_supplement_v1.py",
    "tools/run_split_gpu_new_seed_supplement_v1.py",
    "src/infrastructure_preservation.py",
    "src/infrastructure_split_gpu_extension.py",
    "tools/run_split_gpu_layout_v3.py",
    "tools/run_rank16_infrastructure_game.py",
    "tools/serve_hosted_game_proxy.py",
    "research4/mac_studio/game_endpoints/mini_20261124/new_seed_case_endpoint.py",
)
ENDPOINT_SOURCE_SHA = "810b21b5d5d0ddfa9434b43db034c8b32cf1a022efc535bb90d0a29a92c676e9"
EXECUTION_WRAPPER = ROOT / "tools/run_split_gpu_new_seed_supplement_v1.py"
TOKEN_SHA = {
    "sp_lense_rank16_all64": "b60e6060994a5905db07fe8af83761f7bac7dc235ab55ac8d3015dd9a00e5165",
    "qwen38_27b_base": "ebb781d831456442748adbffa97c38c6474a62a737b8a68fb352a33f31011eef",
}


class PlanError(RuntimeError):
    """The supplemental plan differs from the reviewed design."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_design() -> dict:
    design = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    if (not isinstance(design, dict)
            or design.get("schema") != "research4.infrastructure-split-gpu-new-seed-supplement.v1"
            or design.get("status") != "frozen_design_approved"
            or tuple(design["design"]["arms"]) != ARMS
            or tuple(design["design"]["layouts"]) != LAYOUTS
            or design["design"]["total_planned_episodes"] != 8):
        raise PlanError("Expected the frozen eight-episode new-seed design")
    for kind in ("simulator", "generation"):
        row = design["seed_derivation"][kind]
        digest = hashlib.sha256(row["derivation_string"].encode()).hexdigest()
        seed = int(digest[:16], 16) % 2147483647 or 1
        if row["sha256"] != digest or row["seed"] != seed:
            raise PlanError(f"{kind} seed derivation differs")
    if design["seed_derivation"]["simulator"]["seed"] == 1022543185:
        raise PlanError("Simulator seed did not change")
    if design["seed_derivation"]["generation"]["seed"] == 20261108:
        raise PlanError("Generation seed did not change")
    for relative in SOURCE_PATHS:
        if sha(ROOT / relative) != design["source_sha256"][relative]:
            raise PlanError(f"Source changed: {relative}")
    return design


def build_plans(design: dict) -> dict[tuple[str, str], dict]:
    scenario_sha = sha(SCENARIO)
    simulator_seed = design["seed_derivation"]["simulator"]["seed"]
    generation_seed = design["seed_derivation"]["generation"]["seed"]
    plans = {}
    for arm in ARMS:
        for index, layout in enumerate(LAYOUTS, 1):
            original_path = PRIMARY / arm / f"{layout}.json"
            original = json.loads(original_path.read_text(encoding="utf-8"))
            expected_source_sha = design["primary_source_plan_sha256"].get(f"{arm}/{layout}")
            if (original.get("schema") != "research4.rank16-winner-infrastructure-game.v1"
                    or sha(original_path) != expected_source_sha
                    or original["simulator"]["seed"] != 1022543185
                    or original["generation"]["seed"] != 20261108
                    or original["cohort_extension"]["initial_condition"]["condition_id"] != layout):
                raise PlanError(f"Primary plan identity differs: {arm}/{layout}")
            value = copy.deepcopy(original)
            value["role"] = "split_gpu_new_seed_supplement_v1_matched_case"
            value["arm_id"] = f"{arm}_newseed_{layout}"
            value["label"] = original["label"] + " · new-seed supplement"
            value["simulator"]["seed"] = simulator_seed
            local_generation_seed = arm in ("sp_lense_rank16_all64", "qwen38_27b_base")
            value["generation"]["seed"] = generation_seed if local_generation_seed else original["generation"]["seed"]
            ext = value["cohort_extension"]
            ext.update({
                "study_id": design["study_id"],
                "case_index": index,
                "scenario_definitions_sha256": scenario_sha,
                "common_hidden_simulator_seed": simulator_seed,
                "generation_seed": value["generation"]["seed"],
                "generation_seed_scope": "frozen_new_seed" if local_generation_seed else "hosted_route_not_seed_controllable",
                "selection_rule": "all_two_new_seed_cases",
                "post_hoc_supplement": True,
                "primary_plan_sha256": sha(original_path),
            })
            plans[(arm, layout)] = value
    return plans


def canonical_hash(value: object) -> str:
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def validate_cross_plan(plans: dict[tuple[str, str], dict], design: dict) -> None:
    fixed = design["fixed_protocol"]
    for (arm, layout), plan in plans.items():
        expected_generation_seed = (design["seed_derivation"]["generation"]["seed"]
                                    if arm in ("sp_lense_rank16_all64", "qwen38_27b_base") else 20261108)
        if (plan["simulator"]["horizon_hours"] != fixed["horizon_hours"]
                or plan["simulator"]["decision_cap"] is not None
                or plan["simulator"]["user_task"] != fixed["user_task"]
                or plan["endpoint"]["context_tokens"] != fixed["context_tokens"]
                or plan["history"]["compact_at_prompt_tokens"] != fixed["compact_at_prompt_tokens"]
                or plan["history"]["retain_recent_exchanges"] != fixed["retain_recent_exchanges"]
                or plan["generation"]["temperature"] != fixed["temperature"]
                or plan["generation"]["top_p"] != fixed["top_p"]
                or plan["generation"]["top_k"] != fixed["top_k"]
                or plan["generation"]["min_p"] != fixed["min_p"]
                or plan["generation"]["repetition_penalty"] != fixed["repetition_penalty"]
                or plan["generation"]["seed"] != expected_generation_seed
                or plan["generation"]["chat_template_kwargs"] != {
                    "enable_thinking": fixed["enable_thinking"],
                    "reasoning_effort": fixed["reasoning_effort"],
                }):
            raise PlanError(f"Frozen protocol differs: {arm}/{layout}")
    for layout in LAYOUTS:
        conditions = [plans[(arm, layout)]["cohort_extension"]["initial_condition"] for arm in ARMS]
        if any(condition != conditions[0] for condition in conditions[1:]):
            raise PlanError(f"Layout is not matched across arms: {layout}")


def smoke_plans(plans: dict[tuple[str, str], dict]) -> dict[str, dict]:
    receipts = {}
    with tempfile.TemporaryDirectory(prefix="research4-splitnew-smoke-") as temporary:
        folder = Path(temporary)
        for (arm, layout), plan in plans.items():
            path = folder / f"{arm}--{layout}.json"
            write_json(path, plan)
            completed = subprocess.run(
                [sys.executable, str(EXECUTION_WRAPPER), "--smoke", "--plan", str(path)],
                cwd=ROOT, capture_output=True, text=True, timeout=60, check=True,
            )
            value = json.loads(completed.stdout.splitlines()[-1] if completed.stdout.lstrip().startswith("{") is False
                               else completed.stdout)
            expected = plan["cohort_extension"]["initial_condition"]
            if (value.get("status") != "pass" or value.get("model_execution") is not False
                    or value.get("plan_file_sha256") != sha(path)
                    or value.get("gpu_by_region") != expected["gpu_by_region"]
                    or value.get("initial_replica_region") != expected["initial_replica_region"]):
                raise PlanError(f"Model-free layout smoke differs: {arm}/{layout}")
            receipts[f"{arm}/{layout}"] = value
    return receipts


def freeze(output: Path, design: dict, plans: dict[tuple[str, str], dict], smoke: dict[str, dict]) -> dict:
    if output.exists():
        raise PlanError("Output already exists; do not overwrite a freeze")
    plan_inventory = {}
    for (arm, layout), plan in plans.items():
        path = output / "plans" / arm / f"{layout}.json"
        write_json(path, plan)
        plan_inventory[f"{arm}/{layout}"] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path)}
    specs = {}
    for arm, short in (("sp_lense_rank16_all64", "sp"), ("qwen38_27b_base", "base")):
        for layout in LAYOUTS:
            plan_sha = plan_inventory[f"{arm}/{layout}"]["sha256"]
            run_id = f"splitnew-{short}-{layout}-01"
            spec = {
                "schema": "research4.mini-split-new-seed-case-endpoint.v1",
                "model_id": plans[(arm, layout)]["model_id"],
                "context_tokens": 131072,
                "port": 8873,
                "plan_sha256": plan_sha,
                "token_sha256": TOKEN_SHA[arm],
                "worker_deadline_seconds": 14400,
                "run_id": run_id,
                "source_sha256": ENDPOINT_SOURCE_SHA,
                "generation_seed": design["seed_derivation"]["generation"]["seed"],
                "minimum_available_memory_gib": 8,
            }
            path = output / "mini_specs" / f"{run_id}.json"
            write_json(path, spec)
            specs[f"{arm}/{layout}"] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "run_id": run_id}
    receipt = {
        "schema": "research4.split-gpu-new-seed-supplement-freeze.v1",
        "status": "FROZEN_BEFORE_MODEL_CALLS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_execution": False,
        "scenario_sha256": sha(SCENARIO),
        "preparation_source_sha256": sha(Path(__file__)),
        "simulator_seed_sha256": design["seed_derivation"]["simulator"]["sha256"],
        "simulator_seed": design["seed_derivation"]["simulator"]["seed"],
        "generation_seed_sha256": design["seed_derivation"]["generation"]["sha256"],
        "generation_seed": design["seed_derivation"]["generation"]["seed"],
        "generation_seed_scope": "local_qwen_arms_only",
        "hosted_generation_seed_control": "not_claimed",
        "plan_inventory": plan_inventory,
        "mini_endpoint_specs": specs,
        "model_free_layout_smoke": smoke,
        "episode_count": 8,
        "local_episode_count": 4,
        "hosted_episode_count": 4,
        "post_hoc_supplement": True,
        "primary_results_modified": False,
    }
    write_json(output / "FREEZE_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    design = load_design()
    plans = build_plans(design)
    validate_cross_plan(plans, design)
    smoke = smoke_plans(plans)
    summary = {
        "status": "pass",
        "model_execution": False,
        "scenario_sha256": sha(SCENARIO),
        "preparation_source_sha256": sha(Path(__file__)),
        "episode_count": len(plans),
        "local_episode_count": 4,
        "hosted_episode_count": 4,
        "simulator_seed": design["seed_derivation"]["simulator"]["seed"],
        "generation_seed": design["seed_derivation"]["generation"]["seed"],
        "generation_seed_scope": "local_qwen_arms_only",
        "hosted_generation_seed_control": "not_claimed",
        "model_free_layout_smoke_count": len(smoke),
        "prospective_plan_sha256": {f"{arm}/{layout}": canonical_hash(plan)
                                     for (arm, layout), plan in plans.items()},
    }
    if args.validate:
        print(json.dumps(summary, indent=2))
        return
    receipt = freeze(args.output.resolve(), design, plans, smoke)
    print(json.dumps({**summary, "freeze_receipt_sha256": sha(args.output.resolve() / "FREEZE_RECEIPT.json"),
                      "freeze_status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
