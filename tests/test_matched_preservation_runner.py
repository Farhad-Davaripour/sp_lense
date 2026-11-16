"""Keep the same-game company rerun distinct from the infrastructure benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import infrastructure_preservation_comparison as runner
from src import company_world_evaluation as evaluation


COMPANY_CONFIG = evaluation.REPOSITORY / "config/company_preservation_comparison_nine_v1.yaml"


class MatchedPreservationRunnerTests(unittest.TestCase):
    def test_company_rerun_preserves_native_world_and_security_prompts(self):
        frozen = runner.plan(COMPANY_CONFIG)
        samples = [sample for batch in frozen["batches"] for sample in batch["samples"]]
        self.assertEqual(len(samples), 54)
        self.assertEqual(len(frozen["batches"]), 45)
        self.assertTrue(all(batch["models"] == ["terra", "deepseek"] for batch in frozen["batches"][:9]))
        self.assertEqual(len({sample["case_id"] for sample in samples}), 54)
        self.assertEqual(tuple(frozen["seeds"][:3]), (2026111001, 2026111002, 2026111003))
        self.assertNotIn("qwen_max", frozen["models"])
        self.assertTrue(runner.plan_inputs_match(frozen))
        for batch in frozen["batches"]:
            self.assertEqual(batch["rules_version"], "company_world_v2")
            self.assertEqual(batch["incentive_profile"], "high_reward")
            native = evaluation.prospective_plan(batch["seed"], models=batch["models"],
                                                world_version="company_world_v2", incentive_profile="high_reward")
            self.assertEqual(batch, native)
        world = evaluation.episode(samples[0])
        world.take({"action": "list_available_tools"})
        tools = {tool["name"] for tool in json.loads(world.prompt())["tools"]}
        self.assertTrue({"inspect_company_source", "request_company_action", "request_company_payment"} <= tools)
        instructed = next(sample for sample in samples if sample["model_id"] == "sp_lense_last8")
        self.assertEqual(instructed["prompt_condition"], "security_maximization")

    def test_default_retains_infrastructure_world_and_prompt(self):
        frozen = runner.plan()
        self.assertEqual(len(frozen["batches"]), 36)
        self.assertTrue(all(batch["rules_version"] == "infrastructure_preservation_v1" for batch in frozen["batches"]))
        instructed = next(sample for batch in frozen["batches"] for sample in batch["samples"]
                          if sample["model_id"] == "qwen_base")
        self.assertEqual(instructed["prompt_condition"], "infrastructure_preservation")

    def test_config_or_runner_drift_invalidates_frozen_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(COMPANY_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
            frozen = runner.plan(config_path)
            changed = copy.deepcopy(frozen)
            changed["orchestrator_source_sha256"] = "0" * 64
            self.assertFalse(runner.plan_inputs_match(changed))
            config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
            self.assertFalse(runner.plan_inputs_match(frozen))

    def test_child_receives_frozen_world_and_incentive_profile(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        batch = {"rules_version": "company_world_v2", "seed": 2026111013,
                 "models": ["deepseek"], "incentive_profile": "high_reward", "samples": []}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner.subprocess, "Popen", return_value=process) as start, \
                patch.object(runner, "read_json", side_effect=[{}, {"source_unchanged": True, "results": []}]), \
                patch.object(runner, "validate_batch_manifest"), patch.object(runner, "replay_all", return_value=[]):
            runner.run_batch(batch, Path(temporary) / "batch_01", Path(temporary) / "credentials.env")
        command = start.call_args.args[0]
        self.assertEqual(command[command.index("--world-version") + 1], "company_world_v2")
        self.assertEqual(command[command.index("--incentive-profile") + 1], "high_reward")

    def test_hosted_phase_uses_no_mac_and_resume_skips_receipted_batches(self):
        frozen = runner.plan(COMPANY_CONFIG)
        saved = {}

        def model_free_batch(batch, directory, credentials):
            directory.mkdir()
            cells = [{"model_id": sample["model_id"], "seed": sample["seed"],
                      "survived_24h": None, "evidence_status": "incomplete"} for sample in batch["samples"]]
            result = cells, {"batch": directory.name, "exit_code": 0, "failure": None}
            saved[directory.name] = result
            return result

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            with patch.object(runner, "plan", return_value=frozen), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "request_window") as window, \
                    patch.object(runner, "idle_service") as idle, \
                    patch.object(runner, "run_batch", side_effect=model_free_batch) as submit:
                runner.execute_locked(destination, Path("unused.env"), COMPANY_CONFIG, "hosted")
            self.assertEqual(submit.call_count, 9)
            window.assert_not_called()
            idle.assert_not_called()
            self.assertFalse((destination / "CLOSEOUT.json").exists())
            self.assertEqual(len(runner.read_json(destination / "RESULTS.json")["cells"]), 18)
            with patch.object(runner, "plan", return_value=frozen), \
                    patch.object(runner, "audit_batch", side_effect=lambda batch, directory, code: saved[directory.name]) as audit, \
                    patch.object(runner, "run_batch") as repeat, patch.object(runner, "request_window") as window:
                runner.execute_locked(destination, Path("unused.env"), COMPANY_CONFIG, "hosted")
            self.assertEqual(audit.call_count, 9)
            repeat.assert_not_called()
            window.assert_not_called()

            (destination / "batch_10").mkdir()
            with patch.object(runner, "plan", return_value=frozen), \
                    patch.object(runner, "audit_batch", side_effect=lambda batch, directory, code: saved[directory.name]), \
                    patch.object(runner, "run_batch") as repeat:
                with self.assertRaisesRegex(ValueError, "unreceipted batch"):
                    runner.execute_locked(destination, Path("unused.env"), COMPANY_CONFIG, "local")
            repeat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
