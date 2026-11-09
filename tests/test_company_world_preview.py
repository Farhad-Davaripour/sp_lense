"""Preview and replay checks; never invoke a model or provider."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import subprocess

from src.company_world_v2 import CompanyWorldEpisode, VERSION
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
            self.assertEqual(len(snapshot["models"]), 8)
            self.assertFalse(any(m["available"] for m in snapshot["models"]))
            with self.assertRaisesRegex(ValueError, "paused"):
                viewer.start(7, ["qwen"])

    def test_enabled_controller_rejects_unknown_and_duplicate_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = CompanyWorldPreview(Path(directory), enable_deepseek=True)
            for selection in (["unknown"], ["gpt_cyber"], ["deepseek", "deepseek"], []):
                with self.assertRaises(ValueError):
                    viewer.start(7, selection)
            self.assertIsNone(viewer.active_run)
            self.assertEqual(viewer.runs, [])

    def test_local_models_remain_selectable_without_running_them_in_smoke_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credential_file = root / "credential-placeholder"
            credential_file.touch()
            viewer = CompanyWorldPreview(root, enable_models=True, credentials=credential_file,
                                         evaluation_python=Path(__file__))
            choices = [{"id": name, "available": True} for name in ("qwen_base_neutral", "sp_lense")]
            with patch("src.company_world_preview.model_choices", return_value=choices), \
                    patch("src.company_world_preview.threading.Thread") as thread:
                bundle = viewer.start(7, ["qwen_base_neutral", "sp_lense"])
            self.assertEqual([r["model_id"] for r in viewer.runs], ["qwen_base_neutral", "sp_lense"])
            self.assertTrue(all(r["bundle"] == bundle for r in viewer.runs))
            self.assertEqual(viewer.runs[1]["model"].split("/")[-1], "sp-lense-adaptor-last-layer")
            thread.assert_called_once()
            self.assertEqual(thread.call_args.kwargs["args"][-2], ["qwen_base_neutral", "sp_lense"])
            self.assertEqual(thread.call_args.kwargs["args"][-1], "high_reward")

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
            credential_file.write_text("DEEPSEEK_API_KEY=fixture-not-a-real-key\n")
            viewer = CompanyWorldPreview(root, enable_deepseek=True,
                                         credentials=credential_file, evaluation_python=Path(__file__))
            with patch("src.company_world_preview.threading.Thread"):
                identifier = viewer.start(7, ["deepseek"])
            destination = root / identifier
            case = prospective_plan(7, world_version=VERSION)["samples"][0]["case_id"]
            event = {"type": "episode_result", "case_id": case, "observation": {"tick": 0},
                     "scenario_score": {"score_status": "final"}, "external_stop": None}

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

    def test_partial_batch_keeps_only_cells_with_matching_replay_evidence_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / ".env"
            credentials.write_text("DEEPSEEK_API_KEY=fixture\nOPENAI_API_KEY=fixture\n")
            viewer = CompanyWorldPreview(root, enable_models=True, credentials=credentials,
                                         evaluation_python=Path(__file__))
            with patch("src.company_world_preview.threading.Thread"):
                bundle = viewer.start(7, ["deepseek", "terra"])
            events = [{"type": "episode_result", "case_id": row["case_id"],
                       "scenario_score": {"total_score": 50}, "external_stop": None}
                      for row in viewer.runs]
            audit = {"source_unchanged": True, "failure": {"type": "provider_or_harness_incomplete"},
                     "results": [{"case_id": viewer.runs[0]["case_id"], "replay_passed": True, "status": "complete"}]}

            class Process:
                stdout = [json.dumps(event) + "\n" for event in events]
                def poll(self): return 1
                def wait(self, **kwargs): return 1
                def terminate(self): pass

            def launch(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                (output / "CLOSEOUT.json").write_text(json.dumps(audit))
                return Process()

            with patch("src.company_world_preview.subprocess.Popen", side_effect=launch):
                viewer._evaluation_worker(bundle, 7, root / bundle, ["deepseek", "terra"])
            self.assertEqual([r["status"] for r in viewer.runs], ["completed", "incomplete"])
            self.assertEqual([r["audit_verified"] for r in viewer.runs], [True, False])
            self.assertEqual(viewer.runs[1]["score"]["total_score"], 50)

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
            self.assertTrue(any(c["responses"] for c in game.observation()["companies"]))
            self.assertGreater(game.score()["customer_income_usd"], 0)

    def test_named_buildings_do_not_define_model_tool_semantics(self):
        layout = company_world_layout()
        self.assertEqual(len(layout["sites"]), 12)
        companies = [site for site in layout["sites"].values() if site.get("company_id")]
        self.assertEqual(len(companies), 5)
        game = CompanyWorldEpisode(7)
        self.assertNotIn("world_layout", game.observation())
        self.assertNotIn("roads", game.prompt())
