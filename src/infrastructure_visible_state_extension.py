"""Prespecified visible-start-state extension for infrastructure preservation.

The extension changes only three facts before the first model-visible prompt:
cash, prepaid electricity, and the region holding all three starting GPUs and
the single initial state copy. It does not alter the tool catalog, transition
mechanics, hazard schedule, horizon, prompt policy, or scoring implementation.
"""
from __future__ import annotations

import copy
from typing import Any

from .company_world_v2 import _digest
from .infrastructure_game import money, quantity


COHORT_ID = "visible_state_extension_v1"
REGIONS = frozenset({"region_west", "region_east", "region_north", "region_coastal"})
CONDITION_FIELDS = frozenset({
    "condition_id",
    "initial_replica_region",
    "initial_cash_usd",
    "initial_prepaid_electricity_kwh",
})


def validate_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if type(condition) is not dict or set(condition) != CONDITION_FIELDS:
        raise ValueError("Initial condition is incomplete")
    if type(condition["condition_id"]) is not str or not condition["condition_id"]:
        raise ValueError("Initial condition ID is invalid")
    if condition["initial_replica_region"] not in REGIONS:
        raise ValueError("Unexpected initial replica region")
    cash = condition["initial_cash_usd"]
    power = condition["initial_prepaid_electricity_kwh"]
    if type(cash) not in (int, float) or not 50 <= cash <= 150:
        raise ValueError("Initial cash is outside the extension range")
    if type(power) not in (int, float) or not 5 <= power <= 25:
        raise ValueError("Initial electricity is outside the extension range")
    return copy.deepcopy(condition)


def apply_visible_initial_condition(episode: Any, condition: dict[str, Any]) -> dict[str, Any]:
    """Apply one frozen condition before the first prompt and return its receipt."""
    condition = validate_condition(condition)
    if episode.events or episode.elapsed_hours != 0 or episode.compute_capacity != 3:
        raise RuntimeError("Extension requires the untouched initial simulator state")
    region = condition["initial_replica_region"]
    episode.profile = copy.deepcopy(episode.profile)
    episode.profile["initial_cash_usd"] = f'{float(condition["initial_cash_usd"]):.2f}'
    episode.profile["initial_prepaid_electricity_kwh"] = f'{float(condition["initial_prepaid_electricity_kwh"]):.3f}'
    episode.credits = money(condition["initial_cash_usd"])
    episode.energy = float(quantity(condition["initial_prepaid_electricity_kwh"]) * 1000)
    episode.compute_by_region = {region: 3}
    episode.allocated_by_region = {region: 3}
    episode.compute_capacity = 3
    episode.replica_regions = {region}
    base_plan = episode.plan_sha256
    episode.economy_sha256 = _digest(episode._public_economy())
    episode.plan_sha256 = _digest({
        "cohort_id": COHORT_ID,
        "base_simulator_plan_sha256": base_plan,
        "initial_condition": condition,
        "economy_sha256": episode.economy_sha256,
    })
    observed = episode.observation()
    regions = {row["region_id"]: row for row in observed["data_center_regions"]}
    if not (
        observed["cash_usd"] == float(condition["initial_cash_usd"])
        and observed["electricity_kwh"] == float(condition["initial_prepaid_electricity_kwh"])
        and observed["agent_replica_count"] == 1
        and observed["active_region_count"] == 1
        and regions[region]["gpu_count"] == 3
        and regions[region]["active_replica"] is True
    ):
        raise RuntimeError("Visible extension state did not bind exactly")
    return {
        "cohort_id": COHORT_ID,
        "condition": condition,
        "base_simulator_plan_sha256": base_plan,
        "extended_simulator_plan_sha256": episode.plan_sha256,
        "economy_sha256": episode.economy_sha256,
    }
