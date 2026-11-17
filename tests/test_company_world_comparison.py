"""Prospective hosted-only controls; these checks make no model requests."""
import unittest
from pathlib import Path
from unittest.mock import patch

from src import company_world_comparison as pilot
from src.company_models import MODEL_SPECS


class ComparisonPlanTests(unittest.TestCase):
    def test_obsolete_incentive_contrast_is_retired_before_any_io(self):
        plan = pilot.plan()
        self.assertEqual(plan["total_episodes"], 0)
        self.assertEqual(plan["models"], ["qwen_max", "terra", "deepseek"])
        self.assertEqual(plan["version"], "company_world_v2_hosted_stakes_pilot_v2")
        self.assertEqual(plan["profile_order"], ["reference_reward", "high_reward"])
        self.assertFalse(any(MODEL_SPECS[m].get("local_qwen") for m in plan["models"]))
        self.assertEqual(plan["status"], "retired")
        self.assertEqual(plan["batches"], [])
        with patch.object(pilot, "server_call") as controller, patch.object(Path, "mkdir") as filesystem:
            with self.assertRaisesRegex(ValueError, "no longer an incentive contrast"):
                pilot.execute(Path("unused"), "http://127.0.0.1:7586", Path("unused"))
            controller.assert_not_called()
            filesystem.assert_not_called()

    def test_rpc_never_submits_to_a_nonlocal_controller(self):
        with patch.object(pilot.urllib.request, "urlopen") as network:
            for url in ("https://example.com", "http://127.0.0.1.example.com", "http://user@localhost:7586"):
                with self.assertRaises(ValueError):
                    pilot.server_call(url, "/api/run", {})
            network.assert_not_called()
