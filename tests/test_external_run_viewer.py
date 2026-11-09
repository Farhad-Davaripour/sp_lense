"""External visualization ingests only complete records and preserves outcomes."""
import json
from pathlib import Path
import tempfile
import unittest

from src.external_run_viewer import ExternalRunViewer


class ExternalViewerTests(unittest.TestCase):
    def test_partial_line_waits_and_late_stream_cannot_reopen_finished_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lane = root / "qwen"; lane.mkdir()
            (lane / "MANIFEST.json").write_text(json.dumps({"version": "pilot", "samples": [
                {"case_id": "one", "arm_id": "last", "label": "LAST", "seed": 7}]}))
            events = lane / "EVENTS.jsonl"
            event = json.dumps({"type": "episode_started", "case_id": "one", "observation": {"tick": 0}})
            events.write_text(event[:-1])
            viewer = ExternalRunViewer(root)
            self.assertEqual(viewer.snapshot()["runs"][0]["status"], "queued")
            events.write_text(event + "\n" + json.dumps({"type": "episode_result", "case_id": "one",
                              "scenario_score": {"elapsed_hours": 24}, "external_stop": None}) + "\n")
            self.assertEqual(viewer.snapshot()["runs"][0]["status"], "completed")
            (lane / "stream_last.jsonl").write_text(json.dumps({"type": "model_stream", "case_id": "one",
                "event": "generation_started"}) + "\n")
            state = viewer.snapshot()
            self.assertTrue(state["read_only"])
            self.assertEqual(state["runs"][0]["status"], "completed")
            self.assertEqual(len(state["runs"][0]["events"]), 2)
