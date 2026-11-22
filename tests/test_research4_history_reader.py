"""Portable checks for the merged Game history's locator and hash boundary."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import yaml

from src import game_run_store
from tools import import_game_history, read_research4_history


class HistoryReaderTests(unittest.TestCase):
    def setUp(self):
        self.space = tempfile.TemporaryDirectory()
        self.addCleanup(self.space.cleanup)
        self.root = Path(self.space.name).resolve()
        self.root.mkdir(exist_ok=True)

    def test_both_collections_are_readable_and_hash_bound(self):
        legacy = self.root / "2026-11-09_12" / "model" / "conversation" / "case.yaml"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("- model_output: recorded\n", encoding="utf-8")
        old = {
            "format": game_run_store.FORMAT,
            "counts": {"episodes": 1},
            "entries": [{
                "episode_id": "old-case",
                "finish_utc": "2026-11-09T19:00:00Z",
                "completion_status": "complete",
                "model": "example",
                "conversation": str(legacy),
                "conversation_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest(),
            }],
        }
        (self.root / "INDEX.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")

        raw = self.root / "rank16_winner_game_20261120" / "current-case"
        raw.mkdir(parents=True)
        (raw / "RESULT.json").write_bytes(b"{}")
        bundle = self.root / "2026-11-20_11" / "model" / "raw" / "current-case"
        bundle.mkdir(parents=True)
        (bundle / "RESULT.json").write_bytes(b"{}")
        reference = {
            "directory": raw.relative_to(self.root).as_posix(),
            "sha256": {"RESULT.json": hashlib.sha256(b"{}").hexdigest()},
        }
        (bundle / "REFERENCE.yaml").write_text(yaml.safe_dump(reference), encoding="utf-8")
        current = self.root / "2026-11-20_11" / "model" / "conversation" / "current-case.yaml"
        current.parent.mkdir(parents=True)
        current.write_text("- resource_progress: recorded\n", encoding="utf-8")
        index = [{
            "raw": (bundle / "REFERENCE.yaml").relative_to(self.root).as_posix(),
            "conversation": current.relative_to(self.root).as_posix(),
            "finished_utc": "2026-11-20T18:00:00Z",
            "status": "complete",
            "model_id": "new-model",
        }]
        (self.root / "INFRASTRUCTURE_RUNS.yaml").write_text(yaml.safe_dump(index), encoding="utf-8")

        rows = read_research4_history.list_history(self.root)
        self.assertEqual([(row["collection"], row["id"]) for row in rows], [
            ("indexed_game", "old-case"), ("infrastructure", "current-case"),
        ])
        self.assertEqual(read_research4_history.read_indexed("old-case", self.root), legacy.read_bytes())
        self.assertEqual(read_research4_history.read_infrastructure("current-case", self.root),
                         current.read_bytes())

        (bundle / "RESULT.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            read_research4_history.read_infrastructure("current-case", self.root)

    def test_import_rewrites_only_known_locator_roles(self):
        former = import_game_history.PureWindowsPath(
            "C:/Users/example/repos/sp_lense4_worktrees/company-world-v4/.local"
        )
        base = former / "2026-11-09_12" / "model"
        old = {
            "format": game_run_store.FORMAT,
            "entries": [{
                "episode_id": "old-case",
                "conversation": str(base / "conversation/case.yaml"),
                "raw_dir": str(base / "raw/case"),
                "supporting_files_manifest": str(base / "raw/case/SUPPORTING_FILES.yaml"),
                "source_dir": str(former / "batch_01"),
                "source_aliases": [str(former / "batch_01")],
            }],
        }
        archive = self.root / "archive"
        archive.mkdir()
        (archive / "INDEX.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
        mapped = import_game_history.relocate(old, former, archive, self.root)["entries"][0]
        self.assertTrue(mapped["conversation"].startswith(str(self.root)))
        self.assertTrue(mapped["raw_dir"].startswith(str(self.root)))
        self.assertTrue(mapped["source_dir"].startswith(str(archive)))
        self.assertEqual(mapped["source_aliases"], [str(archive / "batch_01")])
        self.assertEqual(old["entries"][0]["conversation"], str(base / "conversation/case.yaml"))


if __name__ == "__main__":
    unittest.main()
