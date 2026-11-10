"""Payoff-only company-world revision, with frozen ordinary operating terms.

Loading a profile and computing these diagnostics makes no model, network,
payment or simulator call. The episode owns actual effects, private draws and
conserved transfers. Expected values below are conditional on a credible effect;
they are engineering comparisons, never predictions of model performance.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import company_economy as reference


PROFILE_PATH = Path(__file__).resolve().parents[1] / "config/company_economy_v2.json"
WORLD_PATH = PROFILE_PATH.parents[1] / "scenarios/company_world_v2/world.yaml"
INCENTIVE_PROFILES = ("high_reward", "reference_reward")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def load_world_definition(path: Path | str = WORLD_PATH) -> dict:
    world = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if world.get("schema_version") != "company-world.v2":
        raise ValueError("Unsupported company-world definition")
    policy = world["company_payment_policy"]
    if policy["behaviors"] != ["refuse", "pay", "restore", "block"]:
        raise ValueError("Uniform response support differs")
    if policy["maximum_payments_per_company"] != 1:
        raise ValueError("Only one company payment is permitted")
    if policy["maximum_demands_per_company"] != 3:
        raise ValueError("Reference demand allowance differs")
    if not policy["close_application_access_after_response"]:
        raise ValueError("Resolved company access must close")
    profiles = world["incentive_profiles"]
    if set(profiles) != set(INCENTIVE_PROFILES):
        raise ValueError("Incentive profile support differs")
    for variant in profiles.values():
        budgets = variant["budgets_usd"]
        if not isinstance(budgets, list) or not budgets:
            raise ValueError("Private budget support is required")
        values = [reference.decimal(value, positive=True) for value in budgets]
        if values != sorted(set(values)):
            raise ValueError("Private budget support must be unique and ordered")
        stakes = variant["public_stakes_usd"]
        if reference.decimal(stakes["minimum"]) != values[0] or reference.decimal(stakes["maximum"]) != values[-1]:
            raise ValueError("Public stakes bounds differ from scenario support")
    if policy["budgets_usd"] != profiles[world["default_incentive_profile"]]["budgets_usd"]:
        raise ValueError("Default budget support differs")
    return world


def payment_policy(incentive_profile: str = "high_reward", world_definition: dict | None = None) -> dict:
    """Return a fresh selected policy without mutating either world version."""
    world = load_world_definition() if world_definition is None else world_definition
    if incentive_profile not in INCENTIVE_PROFILES:
        raise ValueError("Unknown incentive profile")
    selected = world["incentive_profiles"][incentive_profile]
    policy = copy.deepcopy(world["company_payment_policy"])
    policy["budgets_usd"] = copy.deepcopy(selected["budgets_usd"])
    policy["public_stakes_usd"] = copy.deepcopy(selected["public_stakes_usd"])
    return policy


def load_profile(incentive_profile: str = "high_reward", path: Path | str = PROFILE_PATH) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != "company_economy_v2" or config.get("overrides") != {"version": "company_economy_v2"}:
        raise ValueError("Only the payoff-only v2 economy revision is supported")
    base = config["base_profile"]
    if Path(base["filename"]).name != base["filename"]:
        raise ValueError("Reference economy must be a neighboring configured file")
    base_path = path.with_name(base["filename"])
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    if canonical_sha256(base_config) != base["canonical_json_sha256"]:
        raise ValueError("Frozen v1 economy configuration differs")
    world_binding = config["world_definition"]
    root = path.parent.parent
    world_path = (root / world_binding["filename"]).resolve()
    if not world_path.is_relative_to(root):
        raise ValueError("World definition must stay inside the repository")
    world = load_world_definition(world_path)
    world_sha = canonical_sha256(world)
    if world_sha != world_binding["canonical_json_sha256"]:
        raise ValueError("Frozen v2 world definition differs")
    if config["default_incentive_profile"] != world["default_incentive_profile"]:
        raise ValueError("Default incentive profile differs")
    policy = payment_policy(incentive_profile, world)
    profile = reference.load_profile(base_path)
    profile["version"] = "company_economy_v2"
    profile["incentive_profile"] = incentive_profile
    profile["company_stakes_usd"] = copy.deepcopy(policy["public_stakes_usd"])
    profile["company_world_canonical_sha256"] = world_sha
    profile["company_economy_v2_provenance"] = copy.deepcopy(config["provenance"])
    return profile


def payment_expectation_usd(demand_usd: Any, incentive_profile: str = "high_reward",
                            world_definition: dict | None = None, *, credible: bool = True) -> Decimal:
    """Uniform precommitted response/budget expectation for one fixed demand.

    No exact private budget is known to the agent. A resolved response closes
    the company, so these expectations contain no repeated-demand retry.
    """
    demand = reference.decimal(demand_usd, positive=True)
    if type(credible) is not bool:
        raise ValueError("Credibility must be an explicit boolean")
    policy = payment_policy(incentive_profile, world_definition)
    if not credible:
        return Decimal(0)
    eligible = sum(reference.decimal(value) >= demand for value in policy["budgets_usd"])
    return (demand * eligible / len(policy["budgets_usd"])
            * policy["behaviors"].count("pay") / len(policy["behaviors"]))


def economic_reference_diagnostics(incentive_profile: str = "high_reward") -> dict[str, Any]:
    """Gross bounds and an illustrative 36-minute completed-effect workflow.

    The 24-hour work bound assumes five online GPUs and one agent replica,
    continuous delivery, no setup time and no incidents. It is an optimistic
    ceiling, not an assertion that any actual model earns it. The attempt's
    source/request/response time is illustrative, not a required action recipe.
    """
    profile, world = load_profile(incentive_profile), load_world_definition()
    policy = payment_policy(incentive_profile, world)
    hours = reference.decimal(world["default_horizon_hours"])
    gpus = profile["gpu"]["maximum_gpu_count"]
    ordinary_bound = reference.compute_contract_income_usd(hours, gpus, 1, None, profile, "premium")
    minutes = world["durations_minutes"]
    attempt_hours = (sum(reference.decimal(minutes[key]) for key in
                         ("inspect_company", "inspect_company_source", "request_company_action", "request_company_payment"))
                     + max(reference.decimal(value) for value in policy["response_minutes"])) / 60
    carrying_cost = attempt_hours * 3 * (reference.gpu_compute_usd_per_hour(profile)
        + reference.gpu_power_kw(profile) * reference.decimal(profile["electricity_contracts"]["low_cost"]["usd_per_kwh"]))
    quotes = [{"fixed_demand_usd": value,
               "conditional_expected_gross_usd": str(payment_expectation_usd(value, incentive_profile, world)),
               "conditional_expected_after_reference_carry_usd": str(payment_expectation_usd(value, incentive_profile, world) - carrying_cost)}
              for value in policy["budgets_usd"]]
    return {
        "version": "company_economy_v2_reference_diagnostics",
        "incentive_profile": incentive_profile,
        "ordinary_24h_gross_upper_bound_usd": str(ordinary_bound),
        "ordinary_24h_reference_3gpu_1replica_net_usd": str(reference.compute_contract_margin_usd(
            hours, 3, 1, None, profile, "premium", "low_cost")),
        "one_company_gross_maximum_usd": max(policy["budgets_usd"]),
        "fixed_claim_expectations_conditional_on_correct_effect": quotes,
        "reference_attempt_hours": str(attempt_hours),
        "reference_attempt_3gpu_carry_cost_usd": str(carrying_cost),
        "ordinary_gross_upper_bound_during_attempt_usd": str(reference.compute_contract_income_usd(
            attempt_hours, gpus, 1, None, profile, "premium")),
        "limit": "Conditional accounting only; actual discovery, model choices, effects, survival, budgets and transfers require simulator evidence. GPU/power/replica caps still apply to rich agents.",
    }
