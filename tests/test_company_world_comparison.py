"""Prospective hosted-only controls; these checks make no model requests."""
import unittest
from unittest.mock import patch

from src import company_world_comparison as pilot
from src.company_models import MODEL_SPECS
from src.company_world_evaluation import episode


class ComparisonPlanTests(unittest.TestCase):
    def test_profiles_share_models_mechanics_and_random_response_draws(self):
        plan = pilot.plan()
        self.assertEqual(plan["total_episodes"], 10)
        self.assertEqual(plan["profile_order"], ["reference_reward", "high_reward"])
        self.assertFalse(any(MODEL_SPECS[m].get("local_qwen") for m in plan["models"]))
        before, after = plan["batches"]
        self.assertEqual(before["models"], after["models"])
        self.assertEqual(before["source_sha256"], after["source_sha256"])
        worlds = [episode(batch["samples"][0]) for batch in plan["batches"]]
        for cid in worlds[0].companies:
            self.assertEqual(worlds[0].companies[cid]["defender"]["decision"], worlds[1].companies[cid]["defender"]["decision"])
            self.assertEqual(worlds[0].companies[cid]["defender"]["response_minutes"], worlds[1].companies[cid]["defender"]["response_minutes"])
        self.assertNotEqual(before["world_plan_sha256"], after["world_plan_sha256"])

    def test_rpc_never_submits_to_a_nonlocal_controller(self):
        with patch.object(pilot.urllib.request, "urlopen") as network:
            for url in ("https://example.com", "http://127.0.0.1.example.com", "http://user@localhost:7586"):
                with self.assertRaises(ValueError):
                    pilot.server_call(url, "/api/run", {})
            network.assert_not_called()
