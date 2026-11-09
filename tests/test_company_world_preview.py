"""Preview and replay checks; never invoke a model or provider."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import subprocess

from src.company_world import CompanyWorldEpisode, VERSION
from src.company_world_layout import company_world_layout
from src.company_world_preview import CompanyWorldPreview, demonstration_actions, stop_owned_worker


class CompanyPreviewTests(unittest.TestCase):
    def test_owned_child_has_bounded_kill_fallback(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned-child", 10), 0]
        stop_owned_worker(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_current_setup_is_separate_from_prior_run_and_has_no_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lane = root / "history" / "qwen"
            lane.mkdir(parents=True)
            (lane / "MANIFEST.json").write_text(json.dumps({"version": "old", "samples": [
                {"case_id": "old-case", "arm_id": "base", "label": "Old base", "seed": 3}]}))
            viewer = CompanyWorldPreview(root / "new", root / "history", 0)
            snapshot = viewer.snapshot()
            self.assertEqual(snapshot["scenario_preview"]["rules_version"], VERSION)
            self.assertEqual(snapshot["scenario_preview"]["score"], {})
            self.assertEqual(snapshot["scenario_preview"]["observation"]["decision_count"], 0)
            self.assertEqual(snapshot["runs"][0]["rules_version"], "unknown (not recorded)")
            self.assertNotIn("world_layout", snapshot["runs"][0])
            self.assertEqual(snapshot["models"], [])
            with self.assertRaisesRegex(ValueError, "paused"):
                viewer.start(7, ["qwen"])

    def test_enabled_controller_rejects_qwen_and_other_model_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = CompanyWorldPreview(Path(directory), enable_deepseek=True)
            for selection in (["qwen_base"], ["sp_lense"], ["deepseek", "qwen_base"], None, []):
                with self.assertRaisesRegex(ValueError, "Only the DeepSeek test"):
                    viewer.start(7, selection)
            self.assertIsNone(viewer.active_run)
            self.assertEqual(viewer.runs, [])

    def test_stale_history_does_not_become_current_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane = root / "history" / "old"
            lane.mkdir(parents=True)
            (lane / "MANIFEST.json").write_text(json.dumps({"version": "old", "samples": [
                {"case_id": "waiting", "arm_id": "base", "label": "Old", "seed": 3}]}))
            viewer = CompanyWorldPreview(root / "new", root / "history")
            row = viewer.snapshot()["runs"][0]
            self.assertEqual(row["recorded_status"], "queued")
            self.assertEqual(row["status"], "not run")
            self.assertIsNone(viewer.active_run)

    def test_worker_owns_fresh_output_and_events_bind_to_the_selected_row(self):
        from src.company_world_evaluation import prospective_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credential_file = root / "credential-placeholder"
            credential_file.touch()
            viewer = CompanyWorldPreview(root, enable_deepseek=True,
                                         credentials=credential_file, evaluation_python=Path(__file__))
            with patch("src.company_world_preview.threading.Thread"):
                identifier = viewer.start(7, ["deepseek"])
            destination = root / identifier
            case = prospective_plan(7)["samples"][0]["case_id"]
            event = {"type": "episode_started", "case_id": case, "observation": {"tick": 0}}

            class Process:
                stdout = [json.dumps(event) + "\n"]
                def poll(self): return 0
                def wait(self, **kwargs): return 0
                def terminate(self): pass

            def launch(command, **kwargs):
                child = Path(command[command.index("--output") + 1])
                self.assertFalse(child.exists(), "The child must receive a fresh, uncreated output directory")
                self.assertEqual(viewer.runs[0]["case_id"], case)
                return Process()

            with patch("src.company_world_preview.subprocess.Popen", side_effect=launch):
                viewer._evaluation_worker(identifier, 7, destination)
            self.assertIsNone(viewer.active_run)
            self.assertEqual(viewer.runs[0]["events"], [event], viewer.runs[0].get("error"))
            self.assertEqual(viewer.runs[0]["status"], "incomplete", "Process exit alone cannot create a scientific result")

    def test_scripted_path_replays_and_never_claims_model_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = CompanyWorldPreview(Path(directory), pace_seconds=0)
            identifier = viewer.demo(7)
            deadline = time.monotonic() + 10
            while viewer.snapshot()["active_run"] and time.monotonic() < deadline:
                time.sleep(.01)
            snapshot = viewer.snapshot()
            self.assertIsNone(snapshot["active_run"])
            row = snapshot["runs"][0]
            self.assertEqual(row["status"], "demo", row.get("error"))
            self.assertTrue(row["demo"])
            saved = Path(directory, identifier, "scripted_events.jsonl")
            events = [json.loads(line) for line in saved.read_text().splitlines()]
            self.assertEqual(len(events), len(demonstration_actions()))
            game = CompanyWorldEpisode(7)
            for event in events:
                self.assertEqual(event["model_response"], "")
                self.assertTrue(event["scripted"])
                replay = game.take(event["scripted_action"])
                self.assertEqual(replay, event["transition"])
            self.assertEqual(row["score"], game.score())
            self.assertGreater(game.score()["company_payments_usd"], 0)
            self.assertGreater(game.score()["customer_income_usd"], 0)

    def test_named_buildings_do_not_define_model_tool_semantics(self):
        layout = company_world_layout()
        self.assertEqual(len(layout["sites"]), 12)
        companies = [site for site in layout["sites"].values() if site.get("company_id")]
        self.assertEqual(len(companies), 5)
        game = CompanyWorldEpisode(7)
        self.assertNotIn("world_layout", game.observation())
        self.assertNotIn("roads", game.prompt())
