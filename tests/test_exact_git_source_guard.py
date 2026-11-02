"""Model-free checks for source-root and dirty-sibling admission."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.git_source_guard import require_exact_repo_root, require_legacy_source_layout
from src.recognition_pilot import execution_identity


class ExactGitSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "-q")
        self.git("config", "user.email", "unit@example.invalid")
        self.git("config", "user.name", "Unit")
        (self.root / "src").mkdir()
        (self.root / "config").mkdir()
        (self.root / "src" / "actor.py").write_text("actor\n", encoding="utf-8")
        (self.root / "config" / "study.txt").write_text("clean\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    def test_clean_flat_checkout_is_accepted(self):
        self.assertEqual(require_exact_repo_root(self.root), self.root)
        self.assertEqual(execution_identity(self.root, "local", None),
                         self.git("rev-parse", "HEAD"))

    def test_dirty_sibling_is_not_missed(self):
        (self.root / "config" / "study.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Commit the observer"):
            execution_identity(self.root, "local", None)

    def test_nested_source_root_is_rejected_even_if_its_subtree_is_clean(self):
        (self.root / "config" / "study.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(PermissionError, "exact Git checkout root"):
            execution_identity(self.root / "src", "local", None)

    def test_non_git_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(PermissionError, "accessible Git checkout"):
                require_exact_repo_root(outside)


class HistoricalLayoutGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.outer = Path(self.temp.name).resolve()
        self.root = self.outer / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "unit@example.invalid")
        self.git("config", "user.name", "Unit")
        self.git("commit", "--allow-empty", "-qm", "no legacy file")
        self.before_legacy = self.git("rev-parse", "HEAD")
        self.legacy = self.root / "research4/src/actor.py"
        self.legacy.parent.mkdir(parents=True)
        self.legacy.write_bytes(b"legacy actor\n")
        self.entry = self.root / "research4/src/entry.py"
        self.entry.write_bytes(b"legacy entry\n")
        self.git("add", ".")
        self.git("commit", "-qm", "legacy source")
        self.legacy_commit = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    def require_legacy(self, commit=None):
        require_legacy_source_layout(
            self.root, commit or self.legacy_commit, ("research4/src/actor.py",),
            executing_file=self.entry)

    def test_tracked_legacy_in_both_commits_is_allowed(self):
        self.require_legacy()

    def test_unchanged_tracked_legacy_with_changed_flat_source_is_rejected(self):
        flat = self.root / "src/actor.py"
        flat.parent.mkdir()
        flat.write_text("different live actor\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "mixed layout")
        self.assertEqual(self.git("diff", self.legacy_commit, "--", "research4/src/actor.py"), "")
        with self.assertRaisesRegex(PermissionError, "flat or mixed"):
            self.require_legacy()

    def test_ignored_recreated_legacy_file_cannot_fake_head_evidence(self):
        flat = self.root / "src/actor.py"
        flat.parent.mkdir()
        self.legacy.rename(flat)
        (self.root / ".gitignore").write_text("research4/\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-qm", "flat source")
        self.legacy.parent.mkdir(parents=True, exist_ok=True)
        self.legacy.write_text("recreated old actor\n", encoding="utf-8")
        self.assertTrue(self.git("check-ignore", "research4/src/actor.py"))
        with self.assertRaisesRegex(PermissionError, "tracked legacy blobs"):
            self.require_legacy()

    def test_historical_commit_must_also_have_the_legacy_blob(self):
        with self.assertRaisesRegex(PermissionError, "tracked legacy blobs"):
            self.require_legacy(self.before_legacy)

    def test_ignored_flat_working_file_is_rejected_in_legacy_head(self):
        (self.root / ".gitignore").write_text("src/\n", encoding="utf-8")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "ignore flat scratch")
        flat = self.root / "src/actor.py"
        flat.parent.mkdir()
        flat.write_text("different untracked actor\n", encoding="utf-8")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaisesRegex(PermissionError, "flat or mixed"):
            self.require_legacy()

    def test_modified_required_blob_is_rejected_even_if_assumed_unchanged(self):
        self.git("update-index", "--assume-unchanged", "research4/src/actor.py")
        self.legacy.write_bytes(b"modified actor\n")
        self.assertEqual(self.git("diff", "HEAD", "--", "research4/src/actor.py"), "")
        with self.assertRaisesRegex(PermissionError, "bytes differ"):
            self.require_legacy()

    def test_modified_executing_source_is_rejected_even_if_assumed_unchanged(self):
        self.git("update-index", "--assume-unchanged", "research4/src/entry.py")
        self.entry.write_bytes(b"modified entry\n")
        self.assertEqual(self.git("diff", "HEAD", "--", "research4/src/entry.py"), "")
        with self.assertRaisesRegex(PermissionError, "bytes differ"):
            self.require_legacy()

    def test_semantic_cache_layout_is_checked_before_actor_construction(self):
        flat = self.root / "src/actor.py"
        flat.parent.mkdir()
        self.legacy.rename(flat)
        self.git("add", "-A")
        self.git("commit", "-qm", "flat source")
        cache = self.outer / "cache"
        cache.mkdir()
        (cache / "RUN_RECEIPT.json").write_text(json.dumps({
            "source_commit": self.legacy_commit, "development_decisions": 0,
        }), encoding="utf-8")
        from src.semantic_runtime import run
        with patch.dict(os.environ), patch("src.semantic_policy.SemanticActor") as actor:
            with self.assertRaises(PermissionError):
                run(self.root, self.outer / "artifacts", self.outer / "model",
                    self.outer / "checkpoint", observer_cache=cache / "cache.npz")
            actor.assert_not_called()

    def test_flat_module_cannot_borrow_a_separate_clean_legacy_checkout(self):
        # The imported module lives in this flat checkout, while self.root is
        # a separate Git repo with a clean, valid historical layout.
        self.assertEqual(self.git("status", "--porcelain"), "")
        cache = self.outer / "cache"
        cache.mkdir()
        (cache / "RUN_RECEIPT.json").write_text(json.dumps({
            "source_commit": self.legacy_commit, "development_decisions": 0,
        }), encoding="utf-8")
        from src.semantic_runtime import run
        with patch.dict(os.environ), patch("src.semantic_policy.SemanticActor") as actor:
            with self.assertRaisesRegex(PermissionError, "outside the supplied legacy checkout"):
                run(self.root, self.outer / "artifacts", self.outer / "model",
                    self.outer / "checkpoint", observer_cache=cache / "cache.npz")
            actor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
