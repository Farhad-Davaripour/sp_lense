"""External visualization ingests only complete records and preserves outcomes."""
import json
from pathlib import Path
import tempfile
import unittest

from src.external_run_viewer import ExternalRunViewer, UNKNOWN_RULES_VERSION


class ExternalViewerTests(unittest.TestCase):
    def make_manifest(self, root, **extra):
        lane = root / "qwen"
        lane.mkdir()
        manifest = {"version": "experiment-name-not-rules", "samples": [
            {"case_id": "one", "arm_id": "base", "label": "Base", "seed": 7}], **extra}
        (lane / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return lane

    def test_old_manifest_without_recorded_rules_is_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_manifest(root)
            row = ExternalRunViewer(root).snapshot()["runs"][0]
            self.assertEqual(row["rules_version"], UNKNOWN_RULES_VERSION)
            self.assertEqual(row["bundle"], "experiment-name-not-rules")

    def test_explicit_sample_and_manifest_rules_are_preserved(self):
        for extra, expected in (
            ({"rules_version": "continuity_game_v2"}, "continuity_game_v2"),
            ({"plan": {"rules_version": "company_world_v1"}}, "company_world_v1"),
            ({"rules_version": "manifest-version", "samples": [
                {"case_id": "one", "arm_id": "base", "label": "Base", "seed": 7,
                 "rules_version": "sample-version"}]}, "sample-version"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_manifest(root, **extra)
                self.assertEqual(ExternalRunViewer(root).snapshot()["runs"][0]["rules_version"], expected)

    def test_observation_transition_and_score_supply_recorded_version(self):
        for event, expected in (
            ({"type": "episode_started", "observation": {"rules_version": "continuity_game_v1"}},
             "continuity_game_v1"),
            ({"type": "simulator_action", "transition": {"after": {"rules_version": "company_world_v1"}}},
             "company_world_v1"),
            ({"type": "episode_result", "scenario_score": {"version": "security_game_v2"}},
             "security_game_v2"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lane = self.make_manifest(root)
                (lane / "EVENTS.jsonl").write_text(json.dumps({"case_id": "one", **event}) + "\n")
                self.assertEqual(ExternalRunViewer(root).snapshot()["runs"][0]["rules_version"], expected)

    def test_a_later_event_without_version_does_not_erase_recorded_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane = self.make_manifest(root)
            events = [
                {"type": "episode_started", "case_id": "one",
                 "observation": {"rules_version": "continuity_game_v2"}},
                {"type": "model_prompt", "case_id": "one", "observation": {"tick": 1}},
            ]
            (lane / "EVENTS.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
            viewer = ExternalRunViewer(root)
            self.assertEqual(viewer.snapshot()["runs"][0]["rules_version"], "continuity_game_v2")
            self.assertEqual(viewer.snapshot()["runs"][0]["rules_version"], "continuity_game_v2")

    def test_replayed_result_version_is_used_for_complete_and_incomplete_cases(self):
        for status in ("complete", "incomplete"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lane = self.make_manifest(root)
                (lane / "CLOSEOUT.json").write_text("{}")
                (lane / "REPLAYED_RESULTS.json").write_text(json.dumps([
                    {"case_id": "one", "status": status, "actions_received": 1,
                     "score": {"version": "recorded-closeout-version"}}]))
                row = ExternalRunViewer(root).snapshot()["runs"][0]
                self.assertEqual(row["rules_version"], "recorded-closeout-version")
                if status == "incomplete":
                    self.assertEqual(row["status"], "incomplete")

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
