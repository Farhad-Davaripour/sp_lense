"""Regressions for the working GPU transport and immutable bounded launches."""
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import yaml
from src.config import ROOT
from src.runtime import launcher, runtime_task, bind_launcher
from src.verify import source_digest


class RuntimeTests(unittest.TestCase):
    def test_actual_runtime_cli_accepts_the_heldout_phase(self):
        result = subprocess.run([sys.executable, "-m", "src.runtime", "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("{gate,steering,heldout}", result.stdout)

    def test_launch_values_preserve_shell_metacharacters(self):
        settings = SimpleNamespace(catalog="research", schema="bronze", volume="artifacts",
            volume_root="/Volumes/research/bronze/artifacts", code_path="/Workspace/a b")
        path = "/Volumes/research/bronze/artifacts/a'; echo unwanted; $(bad).json"
        text = launcher(settings, path, "gate").decode()
        exports = dict(shlex.split(line[7:])[0].split("=", 1)
                       for line in text.splitlines() if line.startswith("export "))
        self.assertEqual(exports["RESEARCH3_ADMISSION"], path)
        self.assertEqual(exports["DATABRICKS_CODE_PATH"], settings.code_path)
        with self.assertRaises(ValueError):
            launcher(settings, path, "unknown")

    def test_an_active_job_is_never_mutated_or_rebound(self):
        client = MagicMock()
        client.jobs.list_runs.return_value = [object()]
        with self.assertRaisesRegex(ValueError, "active research job"):
            runtime_task(client, 1)
        with self.assertRaisesRegex(ValueError, "active research job"):
            bind_launcher(client, 1, {}, "/Workspace/unused.sh", b"unused")
        client.jobs.update.assert_not_called()
        client.workspace.import_.assert_not_called()

    def test_source_digest_covers_shell_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            path = root / "src/entry.sh"
            path.write_bytes(b"exit 2\n")
            before = source_digest(root)
            path.write_bytes(b"exit 0\n")
            self.assertNotEqual(source_digest(root), before)

    def test_matched_gpu_tasks_use_the_working_transport_without_recipe_drift(self):
        for name, seconds in [("matched_validation", 2700), ("matched_steering", 18000)]:
            job = yaml.safe_load((ROOT / "resources" / (name + ".job.yml")).read_bytes())["resources"]["jobs"][name]
            tasks = [t for t in job["tasks"] if "ai_runtime_task" in t]
            self.assertEqual(len(tasks), 1)
            self.assertEqual(job["timeout_seconds"], seconds)
            task = tasks[0]
            self.assertNotIn("notebook_task", task)
            self.assertEqual(task["max_retries"], 0)
            compute = task["ai_runtime_task"]["deployments"][0]["compute"]
            self.assertEqual(compute["accelerator_count"], 1)
            spec = next(e["spec"] for e in job["environments"] if e["environment_key"] == task["environment_key"])
            self.assertIn("torch==2.11.0+cu128", spec["dependencies"])


if __name__ == "__main__":
    unittest.main()
