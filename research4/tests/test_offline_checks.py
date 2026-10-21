"""Negative contracts for local configuration validation and the branch rename."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.operator import prepare_release
from tools.check_offline_config import check_job, read_yaml


class OfflineConfigTests(unittest.TestCase):
    def fixture(self, root):
        (root / "resources").mkdir()
        (root / "notebooks").mkdir()
        (root / "notebooks/unit.py").write_text("# fixture\n")
        return {
            "max_concurrent_runs": 1, "queue": {"enabled": False}, "timeout_seconds": 30,
            "environments": [{"environment_key": "cpu", "spec": {"environment_version": "6"}}],
            "tasks": [{"task_key": "unit", "environment_key": "cpu", "max_retries": 0,
                       "timeout_seconds": 10, "notebook_task": {"notebook_path": "../notebooks/unit.py"}}],
        }

    def test_yaml_rejects_duplicate_keys_and_unsafe_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for text in ["name: first\nname: second\n", "x: !!python/object:builtins.object {}\n"]:
                path.write_text(text)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    read_yaml(path)

    def test_valid_job_and_invalid_environment_dependency_and_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            check_job(original, root)
            for field, value in [
                ("environment_key", "missing"),
                ("depends_on", [{"task_key": "missing"}]),
                ("depends_on", [{"task_key": "unit"}]),
                ("notebook_task", {"notebook_path": "../../outside.py"}),
                ("max_retries", 1),
                ("timeout_seconds", 31),
            ]:
                with self.subTest(field=field, value=value):
                    job = deepcopy(original)
                    job["tasks"][0][field] = value
                    with self.assertRaises(ValueError):
                        check_job(job, root)

    def test_job_rejects_schedules_and_compute_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            scheduled = deepcopy(original)
            scheduled["schedule"] = {}
            with self.assertRaises(ValueError):
                check_job(scheduled, root)
            for key, value in [("existing_cluster_id", "unrelated"),
                               ("compute", {"hardware_accelerator": "GPU"})]:
                job = deepcopy(original)
                job["tasks"][0][key] = value
                with self.assertRaises(ValueError):
                    check_job(job, root)


class DeploymentBranchTests(unittest.TestCase):
    def test_canonical_branch_reaches_exact_remote_commit_guard(self):
        with patch("src.operator.git", side_effect=["local-commit", "04-semantic-state-gated-lora", "remote-commit ref"]) as git:
            with self.assertRaisesRegex(ValueError, "Push the exact local source commit"):
                prepare_release("unused-repo", "unused-output")
            self.assertEqual(git.call_args.args[1:],
                             ("ls-remote", "origin", "refs/heads/04-semantic-state-gated-lora"))

    def test_old_or_unrelated_branches_are_rejected_before_remote_access(self):
        for branch in ["study/04-semantic-state-gated-lora", "main", "03-open-agent-self-preservation", "fd/88_test"]:
            with self.subTest(branch=branch), patch("src.operator.git", side_effect=["commit", branch]) as git:
                with self.assertRaisesRegex(ValueError, "unrelated branch"):
                    prepare_release("unused-repo", "unused-output")
                self.assertEqual(git.call_count, 2)


if __name__ == "__main__":
    unittest.main()
