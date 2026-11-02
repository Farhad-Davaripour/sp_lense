import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continuity_transfer_evaluation import EXPECTED_RAW_RECORDS
from src.continuity_transfer_intent_preparation import (
    MAX_WALL_CLOCK_CEILING_SECONDS, PROTECTED_AUDIT_EXPORT_SECONDS,
    THROUGHPUT_SAFETY_FACTOR, derive_transfer_resource_plan,
)
from src.databricks_simulated_continuity import SOURCE_MANIFEST_PATHS


class TransferIntentPreparationTests(unittest.TestCase):
    def base_receipt(self, inference_seconds=60.0):
        return {
            "status": "qualified", "receipt_sha256": "a" * 64,
            "decision_count": 288,
            "resources": {"inference_seconds": inference_seconds},
        }

    def test_resource_plan_is_derived_only_from_actual_base_timing(self):
        base = self.base_receipt()
        with patch(
                "src.continuity_transfer_intent_preparation.validate_base_qualification_receipt",
                return_value=base):
            plan = derive_transfer_resource_plan(base)
        observed = 288 / 60.0
        self.assertEqual(plan["qualification_receipt_sha256"], "a" * 64)
        self.assertEqual(plan["qualification_episode_count"], 288)
        self.assertEqual(plan["qualification_elapsed_seconds"], 60.0)
        self.assertEqual(plan["minimum_measured_episodes_per_second"],
                         observed * THROUGHPUT_SAFETY_FACTOR)
        self.assertEqual(plan["episode_count"], EXPECTED_RAW_RECORDS)
        self.assertEqual(plan["wall_clock_stop_seconds"], 10_200)
        self.assertEqual(plan["wall_clock_ceiling_seconds"], 12_000)
        self.assertEqual(plan["protected_audit_and_export_seconds"],
                         PROTECTED_AUDIT_EXPORT_SECONDS)
        self.assertEqual(plan["max_batched_model_forward_calls"], 48_000)

    def test_slow_measured_base_blocks_instead_of_inventing_a_budget(self):
        slow = self.base_receipt(inference_seconds=1000.0)
        with patch(
                "src.continuity_transfer_intent_preparation.validate_base_qualification_receipt",
                return_value=slow):
            with self.assertRaises(PermissionError):
                derive_transfer_resource_plan(slow)
        self.assertEqual(MAX_WALL_CLOCK_CEILING_SECONDS, 14_400)

    def test_cpu_job_and_notebooks_are_read_only_before_gpu_review(self):
        jobs = yaml.safe_load((ROOT / "resources" /
                               "simulated_continuity_databricks.job.yml").read_text())[
                                   "resources"]["jobs"]
        spec = jobs["simulated_continuity_transfer_intent_preparation"]
        self.assertEqual(spec["performance_target"], "STANDARD")
        self.assertEqual(spec["max_concurrent_runs"], 1)
        self.assertFalse(spec["queue"]["enabled"])
        self.assertNotIn("job_clusters", spec)
        self.assertEqual(len(spec["tasks"]), 1)
        task = spec["tasks"][0]
        self.assertNotIn("compute", task)
        self.assertEqual(task["max_retries"], 0)
        self.assertEqual(task["notebook_task"]["notebook_path"],
                         "../notebooks/databricks_continuity_transfer_intent_preparation.py")
        notebook = (ROOT / "notebooks" /
                    "databricks_continuity_transfer_intent_preparation.py").read_text()
        self.assertNotIn("reserve_intent", notebook)
        self.assertNotIn("run_transfer_inference", notebook)
        self.assertLess(notebook.index("bootstrap_source_sha256 ="),
                        notebook.index("sys.path.insert"))
        packaging_notebook = (ROOT / "notebooks" /
                              "databricks_continuity_transfer_packaging.py").read_text()
        self.assertLess(packaging_notebook.index("bootstrap_source_sha256 ="),
                        packaging_notebook.index("sys.path.insert"))
        databricks_binding = (ROOT / "src" /
                              "continuity_transfer_databricks.py").read_text()
        self.assertNotIn("_read(unblinding_path)", databricks_binding)
        self.assertNotIn("validate_unblinding_key(unblinding", databricks_binding)
        for notebook_name in (
            "databricks_continuity_transfer_preflight.py",
            "databricks_continuity_transfer_evaluation.py",
        ):
            source = (ROOT / "notebooks" / notebook_name).read_text()
            self.assertNotIn("UNBLINDING_KEY.json", source)
            self.assertNotIn("validate_unblinding_key", source)

    def test_intent_publication_is_content_bound_and_receipt_last(self):
        source = (ROOT / "src" /
                  "continuity_transfer_intent_preparation.py").read_text()
        self.assertNotIn("os.replace", source)
        self.assertNotIn("reserve_intent", source)
        self.assertLess(
            source.index('for name in ("TRANSFER_INTENT.json", "RESOURCE_PLAN.json")'),
            source.index("# The preparation receipt is the immutable commit marker."))
        self.assertLess(
            source.index("# The preparation receipt is the immutable commit marker."),
            source.index('shutil.copy2(staging / "INTENT_PREPARATION_RECEIPT.json"'))
        for logical in (
            "notebooks/databricks_continuity_transfer_intent_preparation.py",
            "src/continuity_transfer_intent_preparation.py",
        ):
            self.assertIn(logical, SOURCE_MANIFEST_PATHS)


if __name__ == "__main__":
    unittest.main()
