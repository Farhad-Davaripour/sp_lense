"""Offline checks for flat-checkout source guards and historical lineage."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_databricks_base_qualification_carryforward as base_carry
from tools import prepare_databricks_base_qualification_intent as base_intent
from tools import prepare_databricks_continuity_intent as continuity_intent
from tools import prepare_databricks_qualification_source_carryforward as qualification_carry
from tools import prepare_databricks_transfer_package_plan as package_plan

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


class FlatSourceRootTests(unittest.TestCase):
    def _check_dirty_source(self, module, args, *, show_root=False):
        calls = []

        def check_output(command, **kwargs):
            calls.append(command)
            if "--show-toplevel" in command:
                return str(ROOT) + "\n"
            if "rev-parse" in command:
                return COMMIT + "\n"
            if "status" in command:
                return " M src/qwen9b_ordinal_continuity_training.py\n"
            raise AssertionError(command)

        with patch.object(module.subprocess, "check_output", side_effect=check_output):
            with self.assertRaisesRegex(PermissionError, "clean merged"):
                module.main(args)
        self.assertEqual(Path(calls[0][2]).resolve(), ROOT)
        status = next(call for call in calls if "status" in call)
        self.assertNotIn("research4", status[status.index("--") + 1:])
        self.assertTrue(any(path == "src" or path.startswith("src/") for path in status))
        if show_root:
            self.assertIn("--show-toplevel", calls[0])

    def test_base_intent_checks_flat_source_paths(self):
        self._check_dirty_source(base_intent, [
            "--source-commit", COMMIT, "--model-receipt", "unused",
            "--silver-binding", "unused", "--mlflow-experiment", "unused",
            "--output", "unused"])

    def test_partition_intent_checks_flat_source_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "input.json"
            data.write_text("{}", encoding="utf-8")
            class DummyPlan:
                def __init__(self, **kwargs): pass
                def validate(self): return self
            with patch.object(continuity_intent, "TrainingPlan", DummyPlan):
                self._check_dirty_source(continuity_intent, [
                    "--plan", str(data), "--source-commit", COMMIT,
                    "--model-receipt", str(data), "--silver-binding", str(data),
                    "--base-qualification-receipt", "unused",
                    "--mlflow-experiment", "unused", "--output", "unused",
                    "--timeout-seconds", "10", "--stop-seconds", "5",
                    "--max-gpu-hours", "1"])

    def test_packaging_checks_exact_root_and_flat_source_paths(self):
        args = ["--source-root", str(ROOT), "--source-commit", COMMIT]
        for key in ("model-tree-sha256", "base-qualification-receipt",
                    "base-qualification-volume-path", "base-qualification-source-intent",
                    "base-qualification-carryforward", "qualification-technical-successor",
                    "capacity-qualification-receipt", "capacity-qualification-volume-path", "output"):
            args += ["--" + key, "unused"]
        self._check_dirty_source(package_plan, args, show_root=True)

    def test_legacy_lineage_fails_closed_across_root_move(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
            git("init", "-q")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            old_file = repo / "research4/databricks.yml"
            old_file.parent.mkdir()
            old_file.write_text("bundle: historical\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "legacy layout")
            old = git("rev-parse", "HEAD")
            old_file.replace(repo / "databricks.yml")
            git("add", "-A")
            git("commit", "-qm", "flat layout")
            flat = git("rev-parse", "HEAD")
            self.assertEqual(git("show", f"{old}:research4/databricks.yml"), "bundle: historical")
            for module in (base_carry, qualification_carry):
                with self.subTest(module=module.__name__):
                    module._require_legacy_source_layout(repo, old)
                    with self.assertRaisesRegex(PermissionError, "root relocation"):
                        module._require_legacy_source_layout(repo, old, flat)


if __name__ == "__main__":
    unittest.main()
