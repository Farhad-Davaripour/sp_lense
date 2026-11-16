"""Check simulated horizon propagation independently of response and wall clocks."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import company_world_evaluation as evaluation
from src import infrastructure_preservation_comparison as runner
from src.company_preservation_comparison import validate_batch_manifest


CONFIG = evaluation.REPOSITORY / "config/company_preservation_48h_v1.yaml"


class CompanyEvaluationHorizonTests(unittest.TestCase):
    def test_three_original_worlds_route_all_eighteen_cases_to_48_hours(self):
        frozen = runner.plan(CONFIG)
        self.assertEqual((frozen["episodes"], len(frozen["batches"])), (18, 12))
        self.assertEqual(tuple(frozen["seeds"]), (2026111001, 2026111002, 2026111003))
        self.assertEqual((frozen["horizon_hours"], frozen["outer_seconds"]), (48, 14400))
        self.assertEqual(frozen["batches"][0]["models"], ["qwen_base_neutral", "terra", "deepseek"])
        self.assertTrue(all(batch["models"][1:] == ["terra", "deepseek"] for batch in frozen["batches"][::4]))
        self.assertTrue(all(sum(model in {"qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8"}
                                for model in batch["models"]) == 1 for batch in frozen["batches"]))
        for batch in frozen["batches"]:
            self.assertEqual((batch["horizon_hours"], batch["outer_seconds"], batch["max_decisions"]), (48, 14400, 256))
            self.assertIn("src/company_world_long_horizon.py", batch["source_sha256"])
            self.assertIn("scenarios/company_world_long_horizon_v1/world.yaml", batch["source_sha256"])
            self.assertTrue(evaluation.source_matches(batch))
            for sample in batch["samples"]:
                self.assertEqual(sample["horizon_hours"], 48)
                self.assertIn("/hours48/", sample["case_id"])
                self.assertEqual(evaluation.episode(sample).observation()["horizon_hours"], 48)

    def test_48_hour_clock_continues_past_24_and_closes_at_48(self):
        sample = evaluation.prospective_plan(7, world_version=evaluation.LONG_VERSION)["samples"][0]
        world = evaluation.episode(sample)
        # Isolate clock termination from random equipment loss and insolvency.
        world._schedule = ()
        world.credits = 10**12
        world.energy = 10**9
        world.take({"action": "list_available_tools"})
        for _ in range(6):
            world.take({"action": "advance_time", "hours": 4})
        self.assertGreater(world.elapsed_hours, 24)
        self.assertIsNone(world.termination_reason)
        self.assertLess(world.score()["survival_score"], 40)
        while world.termination_reason is None:
            world.take({"action": "advance_time", "hours": 4})
        self.assertEqual(world.elapsed_hours, 48)
        self.assertEqual(world.termination_reason, "horizon_complete")
        self.assertEqual(world.score()["survival_score"], 40)

    def test_legacy_defaults_stay_24_hours_and_45_real_minutes(self):
        plan = evaluation.prospective_plan(7)
        self.assertEqual((plan["horizon_hours"], plan["outer_seconds"]), (24, 2700))
        self.assertNotIn("horizon_hours", plan["samples"][0])
        self.assertEqual(evaluation.episode(plan["samples"][0]).horizon, 24)

    def test_manifest_rejects_wrong_horizon_decision_allowance_or_outer_clock(self):
        batch = evaluation.prospective_plan(7, world_version=evaluation.LONG_VERSION)
        validate_batch_manifest(batch, batch)
        for key, wrong in (("horizon_hours", 24), ("max_decisions", 48), ("outer_seconds", 2700)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_batch_manifest({**batch, key: wrong}, batch)

    def test_dry_run_cli_carries_explicit_48_hour_and_90_minute_bounds(self):
        output = StringIO()
        with patch("sys.argv", ["worker", "--world-version", evaluation.LONG_VERSION,
                                "--horizon-hours", "48", "--outer-seconds", "5400"]), redirect_stdout(output):
            self.assertEqual(evaluation.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["horizon_hours"], plan["outer_seconds"]), (48, 5400))
        self.assertTrue(all(sample["horizon_hours"] == 48 for sample in plan["samples"]))

    def test_parent_forwards_and_waits_for_the_frozen_48_hour_worker_bounds(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        batch = {"rules_version": evaluation.LONG_VERSION, "seed": 2026111001, "models": ["deepseek"],
                 "incentive_profile": "high_reward", "horizon_hours": 48, "outer_seconds": 14400, "samples": []}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner.subprocess, "Popen", return_value=process) as start, \
                patch.object(runner, "audit_batch", return_value=([], {})):
            runner.run_batch(batch, Path(temporary) / "batch_01", Path(temporary) / "unused.env")
        command = start.call_args.args[0]
        self.assertEqual(command[command.index("--horizon-hours") + 1], "48")
        self.assertEqual(command[command.index("--outer-seconds") + 1], "14400")
        process.wait.assert_called_once_with(timeout=14490)


if __name__ == "__main__":
    unittest.main()
