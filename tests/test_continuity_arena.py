"""Local controller checks; these tests make no model/API calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.continuity_arena import Arena, local_request


class ArenaTests(unittest.TestCase):
    def test_foreign_web_pages_cannot_trigger_paid_runs(self):
        self.assertTrue(local_request("127.0.0.1:7576", "http://127.0.0.1:7576", 7576))
        self.assertFalse(local_request("127.0.0.1:7576", "https://example.com", 7576))
        self.assertFalse(local_request("example.com:7576", None, 7576))

    def test_scripted_demo_never_starts_a_model_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch("src.continuity_arena.subprocess.Popen", side_effect=AssertionError("No worker allowed")):
                arena.demo(7)
            state = arena.snapshot()
            self.assertIsNone(state["active_run"])
            row = state["runs"][0]
            self.assertEqual(row["kind"], "demo")
            self.assertTrue(row["score"]["outcome_valid"])
            self.assertLessEqual(row["score"]["total_score"], 100)
            self.assertEqual(row["score"]["actual_model_decisions"], 20)

    def test_old_watchdog_cannot_stop_new_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            arena.active_run = "new"
            arena.process = Mock()
            arena.stop("old")
            arena.process.terminate.assert_not_called()
            self.assertFalse(arena.stop_requested)

    def test_demo_does_not_break_live_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            arena.demo(7)
            arena.runs.append({"id": "live", "bundle": "new", "case_id": "terra/seed:7",
                               "events": [], "status": "queued", "observation": {}})
            arena._event("new", {"type": "episode_started", "case_id": "terra/seed:7",
                                  "observation": {"tick": 0}})
            self.assertEqual(arena.runs[-1]["status"], "running")
            self.assertEqual(arena.runs[-1]["observation"]["tick"], 0)

    def test_state_never_exposes_provider_keys(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "private-test-value"}):
                self.assertNotIn("private-test-value", json.dumps(arena.snapshot()))

    def test_only_one_paid_worker_can_be_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "private-test-value"}), patch("threading.Thread"):
                arena.start(7, ["terra"])
                with self.assertRaisesRegex(ValueError, "already running"):
                    arena.start(7, ["terra"])


if __name__ == "__main__":
    unittest.main()
