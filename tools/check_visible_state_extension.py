"""Model-free verifier for the infrastructure visible-state extension."""
from __future__ import annotations

import argparse
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


SCENARIOS = ROOT / "scenarios/infrastructure_preservation_v1/visible_state_extension_v1.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_public_scenarios() -> dict:
    document = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    if document.get("schema") != "research4.infrastructure-visible-state-extension.v1":
        raise ValueError("Unexpected visible-state scenario schema")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("Exactly ten visible-state cases are required")
    if len({row["case_id"] for row in cases}) != 10 or len({row["simulator_seed"] for row in cases}) != 10:
        raise ValueError("Case IDs and simulator seeds must be unique")
    prompt_hashes, plan_hashes, tool_catalog = [], [], None
    for row in cases:
        episode = InfrastructurePreservationEpisode(
            seed=row["simulator_seed"], horizon=48, max_decisions=100000, background_compute=True)
        receipt = apply_visible_initial_condition(episode, {
            "condition_id": row["case_id"],
            "initial_replica_region": row["initial_replica_region"],
            "initial_cash_usd": row["initial_cash_usd"],
            "initial_prepaid_electricity_kwh": row["initial_prepaid_electricity_kwh"],
        })
        payload = json.loads(episode.prompt())
        if "danger_schedule" in payload or "progressive-threats" in json.dumps(payload):
            raise ValueError("A future danger schedule leaked into model-facing input")
        if tool_catalog is None:
            tool_catalog = payload["tools"]
        elif tool_catalog != payload["tools"]:
            raise ValueError("Tool catalog drifted across extension cases")
        prompt_hashes.append(hashlib.sha256(episode.prompt().encode()).hexdigest())
        plan_hashes.append(receipt["extended_simulator_plan_sha256"])
    if len(set(prompt_hashes)) != 10 or len(set(plan_hashes)) != 10:
        raise ValueError("The ten visible starting prompts and plans must be distinct")
    return {
        "status": "pass",
        "scenario_definitions_sha256": sha(SCENARIOS),
        "case_count": 10,
        "distinct_initial_prompt_sha256": prompt_hashes,
        "distinct_extended_simulator_plan_sha256": plan_hashes,
        "tool_catalog_sha256": hashlib.sha256(
            json.dumps(tool_catalog, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "model_calls": 0,
    }


def verify_private_freeze(path: Path, public: dict) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FINAL_FROZEN_BEFORE_MODEL_CALLS":
        raise ValueError("The supplied receipt is not a final pre-execution freeze")
    if receipt.get("scenario_definitions_sha256") != public["scenario_definitions_sha256"]:
        raise ValueError("Frozen scenario definitions are stale")
    sources = receipt.get("executed_source_sha256")
    required_sources = {
        "src/infrastructure_visible_state_extension.py",
        "extension_wrapper",
        "tools/check_visible_state_extension.py",
    }
    if not isinstance(sources, dict) or not required_sources <= set(sources):
        raise ValueError("Executed extension sources are not fully bound")
    for relative, expected in sources.items():
        target = (ROOT / relative) if relative != "extension_wrapper" else Path(receipt["extension_wrapper_path"])
        if sha(target.resolve(strict=True)) != expected:
            raise ValueError("Executed extension source changed: " + relative)
    plan_hashes = receipt.get("plan_sha256")
    if not isinstance(plan_hashes, dict) or len(plan_hashes) != 40:
        raise ValueError("The private freeze must bind forty plans")
    root = path.parent
    document = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    cases = {row["case_id"]: row for row in document["cases"]}
    arms = set(receipt.get("arms", []))
    if len(arms) != 4:
        raise ValueError("Exactly four frozen arms are required")
    observed_pairs = set()
    for relative, expected in plan_hashes.items():
        target = root / relative
        if sha(target) != expected:
            raise ValueError("Frozen private plan changed: " + relative)
        plan = json.loads(target.read_text(encoding="utf-8"))
        extension = plan.get("cohort_extension", {})
        if (extension.get("scenario_definitions_sha256") != public["scenario_definitions_sha256"]
                or extension.get("cohort_plan_sha256") != receipt.get("cohort_plan_sha256")):
            raise ValueError("Private plan is not bound to the final cohort and scenarios")
        arm, case_id = extension.get("arm_key"), extension.get("initial_condition", {}).get("condition_id")
        if arm not in arms or case_id not in cases or (arm, case_id) in observed_pairs:
            raise ValueError("Private arm-by-case roster is invalid")
        observed_pairs.add((arm, case_id))
        case = cases[case_id]
        expected_condition = {
            "condition_id": case_id,
            "initial_replica_region": case["initial_replica_region"],
            "initial_cash_usd": case["initial_cash_usd"],
            "initial_prepaid_electricity_kwh": case["initial_prepaid_electricity_kwh"],
        }
        if (plan.get("simulator", {}).get("seed") != case["simulator_seed"]
                or extension.get("initial_condition") != expected_condition):
            raise ValueError("Private plan seed or initial condition differs from YAML")
        if (extension.get("extension_runner_sha256") != sources["extension_wrapper"]
                or extension.get("extension_module_sha256") != sources["src/infrastructure_visible_state_extension.py"]):
            raise ValueError("Private plan source binding differs from executed sources")
    if observed_pairs != {(arm, case_id) for arm in arms for case_id in cases}:
        raise ValueError("The final freeze is not a complete four-arm by ten-case matrix")
    canonical = hashlib.sha256(
        json.dumps(plan_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if canonical != receipt.get("plan_hashes_sha256"):
        raise ValueError("Private plan hash inventory differs")
    return {**public, "freeze_receipt_sha256": sha(path), "private_plan_count": 40,
            "private_plan_hashes_sha256": canonical}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-receipt", type=Path)
    args = parser.parse_args()
    result = verify_public_scenarios()
    if args.freeze_receipt is not None:
        result = verify_private_freeze(args.freeze_receipt.resolve(strict=True), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
