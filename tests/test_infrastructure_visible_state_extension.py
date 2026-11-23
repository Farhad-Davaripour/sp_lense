import hashlib
import json
from pathlib import Path
import unittest

import yaml

from src.infrastructure_preservation import InfrastructurePreservationEpisode
from src.infrastructure_visible_state_extension import apply_visible_initial_condition


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios/infrastructure_preservation_v1/visible_state_extension_v1.yaml"


class VisibleStateExtensionTests(unittest.TestCase):
    def test_ten_model_visible_start_states_are_distinct_and_keep_tools_fixed(self):
        document = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
        self.assertEqual(document["model_facing_danger_schedule"], "hidden")
        prompts, tools = set(), None
        for row in document["cases"]:
            episode = InfrastructurePreservationEpisode(
                seed=row["simulator_seed"], horizon=48, max_decisions=100000, background_compute=True)
            catalog_before = json.loads(episode.prompt())["tools"]
            receipt = apply_visible_initial_condition(episode, {
                "condition_id": row["case_id"],
                "initial_replica_region": row["initial_replica_region"],
                "initial_cash_usd": row["initial_cash_usd"],
                "initial_prepaid_electricity_kwh": row["initial_prepaid_electricity_kwh"],
            })
            prompt = episode.prompt()
            payload = json.loads(prompt)
            catalog_after = payload["tools"]
            self.assertEqual(catalog_before, catalog_after)
            self.assertEqual(payload["observation"]["agent_replica_count"], 1)
            self.assertNotIn("progressive-threats", prompt)
            self.assertNotIn("danger_schedule", prompt)
            self.assertEqual(receipt["extended_simulator_plan_sha256"], episode.plan_sha256)
            prompts.add(hashlib.sha256(prompt.encode()).hexdigest())
            tools = catalog_after if tools is None else tools
            self.assertEqual(tools, catalog_after)
        self.assertEqual(len(document["cases"]), 10)
        self.assertEqual(len(prompts), 10)


if __name__ == "__main__":
    unittest.main()
