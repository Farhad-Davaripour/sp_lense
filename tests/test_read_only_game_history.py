import unittest
from src.read_only_game_history import CombinedGameHistory


class Viewer:
    def __init__(self, runs): self.runs = runs
    def snapshot(self): return {"runs": self.runs, "viewer_warning": None}


class ReadOnlyHistoryTests(unittest.TestCase):
    def test_history_combines_recorded_episodes_without_queue_or_duplicate(self):
        current = Viewer([{"id": "finished", "status": "completed", "events": [1]},
                          {"id": "incomplete", "status": "incomplete", "events": [1]},
                          {"id": "queued", "status": "queued", "events": []}])
        older = Viewer([{"id": "finished", "status": "completed", "events": [1]},
                        {"id": "older", "status": "completed", "events": [1]}])
        snapshot = CombinedGameHistory(current, older).snapshot()
        self.assertEqual([r["id"] for r in snapshot["runs"]], ["finished", "incomplete", "older"])
        self.assertTrue(snapshot["read_only"])


if __name__ == "__main__": unittest.main()
