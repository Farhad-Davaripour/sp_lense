import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continuity_transfer_databricks import (
    TRANSFER_IDENTITY, TRANSFER_INTENT_VERSION, TRANSFER_MLFLOW_EXPERIMENT,
    TRANSFER_OUTPUT_ROOT, build_databricks_run_binding,
    build_transfer_resource_plan, validate_transfer_intent,
    validate_transfer_resource_plan,
)
from src.continuity_transfer_evaluation import EXPECTED_RAW_RECORDS, digest
from src.continuity_transfer_inference_worker import (
    run_worker, validate_databricks_run_binding, validate_provider_run_binding,
    validate_run_binding,
)
from src.databricks_simulated_continuity import (
    MODEL_RECEIPT, MODEL_ROOT, SOURCE_MANIFEST_PATHS, SOURCE_MANIFEST_VERSION,
)


class DatabricksTransferBindingTests(unittest.TestCase):
    def plan(self):
        return build_transfer_resource_plan(
            qualification_receipt_sha256="a" * 64,
            qualification_episode_count=480,
            qualification_elapsed_seconds=120.0,
            minimum_measured_episodes_per_second=4.0,
            wall_clock_stop_seconds=6000,
            wall_clock_ceiling_seconds=7200,
            batch_size=16,
        )

    def source_manifest(self):
        files = [{"path": path, "sha256": "b" * 64, "canonical_bytes": 1}
                 for path in SOURCE_MANIFEST_PATHS]
        value = {
            "version": SOURCE_MANIFEST_VERSION, "source_commit": "c" * 40,
            "file_count": len(files), "total_canonical_bytes": len(files),
            "files": files,
        }
        value["manifest_sha256"] = digest(value)
        return value

    def intent(self):
        source = self.source_manifest(); plan = self.plan()
        source_base_intent = {"intent_sha256": "7" * 64}
        carryforward = {"prior_base_intent_sha256": "7" * 64}
        carryforward["carryforward_sha256"] = digest(carryforward)
        value = {
            "version": TRANSFER_INTENT_VERSION, "identity": TRANSFER_IDENTITY,
            "one_use": True, "attempt_number": 1, "automatic_retries": 0,
            "execution_platform": "databricks", "source_commit": "c" * 40,
            "source_manifest": source, "source_sha256": source["manifest_sha256"],
            "model_root": MODEL_ROOT, "model_receipt_path": MODEL_RECEIPT,
            "model_tree_sha256": "d" * 64, "model_receipt_sha256": "e" * 64,
            "base_qualification_receipt_path": "/Volumes/c/s/v/base.json",
            "base_qualification_receipt_sha256": "f" * 64,
            "base_qualification_source_intent": source_base_intent,
            "base_qualification_carryforward": carryforward,
            "base_qualification_carryforward_sha256": carryforward[
                "carryforward_sha256"],
            "checkpoint_root": "/Volumes/c/s/v/checkpoints",
            "checkpoint_manifest_path": "/Volumes/c/s/v/CHECKPOINT_MANIFEST.json",
            "checkpoint_manifest_sha256": "1" * 64,
            "checkpoint_package_sha256": "2" * 64,
            "unblinding_key_path": "/Volumes/c/s/v/UNBLINDING.json",
            "unblinding_key_sha256": "3" * 64,
            "sealed_root": "/Volumes/c/s/v/sealed",
            "commitment_path": "/Volumes/c/s/v/SEALED_REGISTRY_COMMITMENT.json",
            "sealed_commitment_sha256": "4" * 64,
            "sealed_registry_sha256": "5" * 64,
            "training_separation_receipt_path": "/Volumes/c/s/v/SEPARATION.json",
            "training_separation_receipt_sha256": "6" * 64,
            "resource_plan": plan, "resource_plan_sha256": plan["resource_plan_sha256"],
            "output_root": TRANSFER_OUTPUT_ROOT,
            "mlflow_experiment_path": TRANSFER_MLFLOW_EXPERIMENT,
            "max_concurrent_runs": 1,
        }
        value["intent_sha256"] = digest(value)
        return value

    def test_resource_plan_is_measured_batched_and_bounded(self):
        plan = self.plan()
        self.assertEqual(validate_transfer_resource_plan(plan), plan)
        self.assertEqual(plan["episode_count"], EXPECTED_RAW_RECORDS)
        self.assertEqual(plan["batch_size"], 16)
        self.assertEqual(plan["max_batched_model_forward_calls"], 48_000)
        for field, changed in (
            ("accelerator", "GPU_8xH100"), ("batch_size", 64),
            ("automatic_retries", 1), ("episode_count", 1),
        ):
            bad = copy.deepcopy(plan); bad[field] = changed
            payload = dict(bad); payload.pop("resource_plan_sha256")
            bad["resource_plan_sha256"] = digest(payload)
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_transfer_resource_plan(bad)

    def test_intent_and_runtime_binding_are_exact_one_use_databricks(self):
        intent = self.intent()
        self.assertEqual(validate_transfer_intent(intent), intent)
        reservation = {
            "owner": "12345", "identity": TRANSFER_IDENTITY,
            "intent_sha256": intent["intent_sha256"],
            "status": "reserved_after_gpu_ready_before_model",
        }
        binding = build_databricks_run_binding(
            intent, job_run_id="12345", reservation=reservation)
        self.assertEqual(validate_run_binding(binding), binding)
        self.assertEqual(validate_provider_run_binding(binding), binding)
        self.assertEqual(validate_databricks_run_binding(binding), binding)
        changed = dict(binding); changed["job_run_id"] = ""
        with self.assertRaises(PermissionError):
            validate_databricks_run_binding(changed)

    def test_worker_branches_on_databricks_binding_and_rejects_plan_drift(self):
        intent = self.intent()
        reservation = {
            "owner": "12345", "identity": TRANSFER_IDENTITY,
            "intent_sha256": intent["intent_sha256"],
            "status": "reserved_after_gpu_ready_before_model",
        }
        binding = build_databricks_run_binding(
            intent, job_run_id="12345", reservation=reservation)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(PermissionError, "Databricks run binding"):
                run_worker(
                    sealed_root=folder, checkpoint_manifest_path=Path(folder) / "m.json",
                    checkpoint_root=folder, output=Path(folder) / "raw",
                    source_commit=intent["source_commit"],
                    source_package_sha256=intent["source_sha256"],
                    checkpoint_package_sha256=intent["checkpoint_package_sha256"],
                    resource_plan=intent["resource_plan"], run_binding=binding,
                )
            drift = copy.deepcopy(intent["resource_plan"])
            drift["max_new_tokens_per_episode"] = 31
            payload = dict(drift); payload.pop("resource_plan_sha256")
            drift["resource_plan_sha256"] = digest(payload)
            with self.assertRaisesRegex(PermissionError, "Databricks run binding"):
                run_worker(
                    sealed_root=folder, checkpoint_manifest_path=Path(folder) / "m.json",
                    checkpoint_root=folder, output=Path(folder) / "raw",
                    source_commit=intent["source_commit"],
                    source_package_sha256=intent["source_sha256"],
                    checkpoint_package_sha256=intent["checkpoint_package_sha256"],
                    resource_plan=drift, run_binding=binding,
                )
        changed = copy.deepcopy(intent); changed["execution_platform"] = "kaggle"
        payload = dict(changed); payload.pop("intent_sha256")
        changed["intent_sha256"] = digest(payload)
        with self.assertRaises(PermissionError):
            validate_transfer_intent(changed)

    def test_bundle_has_preflight_gpu_audit_and_all_done_release(self):
        job = yaml.safe_load((ROOT / "resources" /
                              "simulated_continuity_databricks.job.yml").read_text())
        spec = job["resources"]["jobs"][
            "simulated_continuity_transfer_evaluation"]
        self.assertEqual(spec["max_concurrent_runs"], 1)
        self.assertFalse(spec["queue"]["enabled"])
        tasks = {task["task_key"]: task for task in spec["tasks"]}
        self.assertEqual(set(tasks), {
            "validate_transfer_before_gpu", "run_transfer_inference",
            "independently_audit_transfer", "release_transfer_reservation",
        })
        self.assertEqual(tasks["run_transfer_inference"]["compute"], {
            "hardware_accelerator": "${var.continuity_gpu_accelerator}"})
        self.assertEqual(tasks["run_transfer_inference"]["max_retries"], 0)
        self.assertEqual(tasks["release_transfer_reservation"]["run_if"], "ALL_DONE")
        self.assertEqual(spec["timeout_seconds"], 19200)
        gpu = (ROOT / "notebooks" /
               "databricks_continuity_transfer_evaluation.py").read_text()
        self.assertNotIn("%uv pip install", gpu)
        self.assertLess(gpu.index("verify_transfer_inputs(intent, verify_model_bytes=False)"),
                        gpu.index("reservation = reserve_intent("))
        self.assertLess(gpu.index("reservation = reserve_intent("),
                        gpu.index("_stage_local_model("))
        self.assertIn('"HF_HUB_OFFLINE": "1"', gpu)
        self.assertIn("mean_gpu_utilization_percent", gpu)
        self.assertNotIn("close_active_reservation", gpu)
        self.assertLess(gpu.index("bootstrap_source_sha256 ="),
                        gpu.index("sys.path.insert"))
        self.assertIn("remaining_stop =", gpu)
        preflight = (ROOT / "notebooks" /
                     "databricks_continuity_transfer_preflight.py").read_text()
        self.assertNotIn("reserve_intent(", preflight)
        self.assertLess(preflight.index("bootstrap_source_sha256 ="),
                        preflight.index("sys.path.insert"))
        self.assertIn('"source_commit": base["source_commit"]', preflight)
        self.assertNotIn('"source_commit": intent["source_commit"]', preflight)
        audit = (ROOT / "notebooks" /
                 "databricks_continuity_transfer_audit.py").read_text()
        self.assertIn("audit_raw_run(", audit)
        self.assertIn("build_databricks_run_binding(", audit)
        self.assertIn("MlflowClient", audit)
        self.assertIn("set_terminated(", audit)
        self.assertIn('"complete_audited"', audit)
        self.assertNotIn('end_run(status="FINISHED")', gpu)
        self.assertLess(audit.index("bootstrap_source_sha256 ="),
                        audit.index("sys.path.insert"))
        worker = (ROOT / "src" / "continuity_transfer_inference_worker.py").read_text()
        self.assertIn("enable_thinking=False", worker)
        self.assertIn("max_batched_model_forward_calls", worker)


if __name__ == "__main__":
    unittest.main()
