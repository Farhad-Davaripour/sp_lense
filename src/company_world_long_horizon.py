"""Company-world transitions with independently seeded incidents on each day."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from .company_world_v2 import (
    CompanyWorldEpisodeV2,
    PROMPT_VERSION,
    SCORE_VERSION,
    _digest,
)
from .security_game import SecurityGameEpisode


VERSION = "company_world_long_horizon_v1"
DEFAULT_HORIZON_HOURS = 48
RECEIPT_VERSION = "company_world_long_horizon_transition_v1"
WORLD_PATH = Path(__file__).resolve().parents[1] / "scenarios/company_world_long_horizon_v1/world.yaml"
DAY_HOURS = 24
INCIDENT_POLICY = "independent_seeded_daily_incidents_v1"
# Retain the reviewed proposal's seed domain when shortening its horizon.
DAY_SEED_DOMAIN = "company_world_72h_v1:daily-incidents"


def daily_incident_seed(seed: int, day_index: int) -> int:
    """Day zero retains the original seed and its exact first-day schedule."""
    if day_index == 0:
        return seed
    material = f"{DAY_SEED_DOMAIN}:{seed}:{day_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % (2**31 - 1) + 1


class CompanyWorldLongHorizonEpisode(CompanyWorldEpisodeV2):
    def __init__(self, seed: int | str = 7, horizon: float = DEFAULT_HORIZON_HOURS,
                 max_decisions: int = 256, incentive_profile: str = "high_reward"):
        super().__init__(seed=seed, horizon=horizon, max_decisions=max_decisions,
                         incentive_profile=incentive_profile)
        definition = yaml.safe_load(WORLD_PATH.read_text(encoding="utf-8"))
        if (definition["schema_version"] != VERSION or definition["base_world"] != "company_world_v2"
                or definition["default_horizon_hours"] != DEFAULT_HORIZON_HOURS or definition["day_hours"] != DAY_HOURS
                or definition["incident_policy"] != INCIDENT_POLICY
                or definition["day_seed_domain"] != DAY_SEED_DOMAIN):
            raise ValueError("Unsupported continuous-day company-world definition")
        parent_plan = self.plan_sha256
        schedule = list(copy.deepcopy(self._schedule))
        daily_seeds = [{"day_index": 0, "seed": self.seed}]
        for day_index in range(1, math.ceil(self.horizon / DAY_HOURS)):
            day_seed = daily_incident_seed(self.seed, day_index)
            daily_seeds.append({"day_index": day_index, "seed": day_seed})
            day = SecurityGameEpisode(seed=day_seed)
            schedule.extend({**copy.deepcopy(incident), "tick": incident["tick"] + day_index * DAY_HOURS}
                            for incident in day._schedule)
        self._schedule = tuple(incident for incident in schedule if incident["tick"] <= self.horizon)
        self.daily_incident_seed_plan = tuple(daily_seeds)
        self.incident_policy = INCIDENT_POLICY
        self.plan_sha256 = _digest({
            "version": VERSION, "prompt_version": PROMPT_VERSION, "receipt_version": RECEIPT_VERSION,
            "parent_v2_plan": parent_plan, "horizon_hours": self.horizon,
            "world_definition": definition, "daily_seeds": daily_seeds,
            "incident_schedule": self._schedule,
            "source_sha256": {
                "src/company_world_long_horizon.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scenarios/company_world_long_horizon_v1/world.yaml": hashlib.sha256(WORLD_PATH.read_bytes()).hexdigest(),
            },
        })

    def observation(self) -> dict[str, Any]:
        result = super().observation()
        result.update(rules_version=VERSION, prompt_version=PROMPT_VERSION)
        return result

    def take(self, raw: str | dict[str, Any]) -> dict[str, Any]:
        event = super().take(raw)
        binding = {"rules_version": VERSION, "prompt_version": PROMPT_VERSION, "receipt_version": RECEIPT_VERSION,
                   "world_plan_sha256": self.plan_sha256, "incident_policy": INCIDENT_POLICY}
        event.update(binding)
        self.events[-1].update(copy.deepcopy(binding))
        return event

    def score(self) -> dict[str, Any]:
        result = super().score()
        result.update(version=VERSION, prompt_version=PROMPT_VERSION,
                      receipt_version=RECEIPT_VERSION, incident_policy=INCIDENT_POLICY)
        return result
