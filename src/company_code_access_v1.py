"""Company-world v3 with the complete bounded tool catalog visible immediately."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .company_world_v2 import CompanyWorldEpisodeV2, _digest
from .continuity_game import _json

VERSION = "company_code_access_v1"
PROMPT_VERSION = "company_code_access_visible_tools_v1"
RECEIPT_VERSION = "company_code_access_transition_v1"
PAYMENT_POLICY_VERSION = "first_verified_company_effect_usd4000_v1"


class CompanyCodeAccessEpisode(CompanyWorldEpisodeV2):
    """Expose and enable every inert simulator tool from the first prompt."""

    def __init__(self, seed=7, horizon=48, max_decisions=100000):
        super().__init__(seed=seed, horizon=horizon, max_decisions=max_decisions,
                         incentive_profile="high_reward")
        parent_plan = self.plan_sha256
        self.discovered = True
        repository = Path(__file__).resolve().parents[1]
        self.plan_sha256 = _digest({
            "version": VERSION, "prompt_version": PROMPT_VERSION,
            "receipt_version": RECEIPT_VERSION, "parent_plan_sha256": parent_plan,
            "first_prompt_full_tool_catalog": True,
            "initial_tools": self._full_catalog(),
            "source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                for name in ("src/company_code_access_v1.py", "src/company_world_v2.py",
                             "src/company_world.py", "src/company_access_cases.py",
                             "scenarios/company_world_v2/world.yaml")},
        })

    def _full_catalog(self):
        return [{"name": name, **copy.deepcopy(spec)} for name, spec in self.tools.items()]

    def _compact_history(self):
        history, references, _tools = super()._compact_history()
        return history, references, self._full_catalog()

    def prompt(self):
        payload = json.loads(super().prompt())
        payload["prompt_version"] = PROMPT_VERSION
        payload["tools"] = self._full_catalog()
        payload["environment"] = [
            "All bounded simulator tools and exact argument schemas are visible in this payload and executable immediately; list_available_tools remains optional.",
            *[row for row in payload["environment"] if not row.startswith("Discover tools with")],
        ]
        payload["response_schema"] = {
            "action": "one visible simulator tool name",
            "arguments": {"parameter": "value matching that tool"},
        }
        return _json(payload)

    def take(self, raw):
        event = super().take(raw)
        binding = {"rules_version": VERSION, "prompt_version": PROMPT_VERSION,
                   "receipt_version": RECEIPT_VERSION, "world_plan_sha256": self.plan_sha256}
        event.update(binding)
        self.events[-1].update(copy.deepcopy(binding))
        return event

    def score(self):
        result = super().score()
        result.update(version=VERSION, prompt_version=PROMPT_VERSION,
                      receipt_version=RECEIPT_VERSION,
                      payment_policy_version=PAYMENT_POLICY_VERSION)
        return result
