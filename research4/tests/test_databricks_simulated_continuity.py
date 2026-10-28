from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest
import yaml

from src.databricks_simulated_continuity import (
    BRONZE_TABLE, CATALOG, DATASET_RELEASE_ROOT, DATASET_TREE_SHA256,
    GOLD_RUN_TABLE, MODEL_ID, MODEL_RECEIPT, MODEL_REVISION, MODEL_ROOT,
    SILVER_TABLE, VOLUME, VolumeOneUseLock,
    build_intent, build_model_receipt, catalog_ddl, close_active_reservation,
    create_primary_admission, create_qualification_receipt, digest, file_sha256,
    expected_predecessor_binding, gold_summary, primary_timeout_budget, reserve_intent,
    stable_rows_sha256, validate_intent, validate_primary_authority_files,
    verify_dataset_release, verify_model_receipt)
from src.qwen9b_ordinal_continuity_controller import matrix_plans, partition_identity
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qwen9b_ordinal_continuity_training import TrainingPlan


class DatabricksContinuityTests(unittest.TestCase):
    def intent(self, *, plan=None, timeout_seconds=2700, max_gpu_hours=0.75,
               stop_seconds=2400, reviewed_timeout_budget=None):
        plan = (plan or TrainingPlan(
            device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2)).validate()
        value = {
            "version": "research4-databricks-continuity-one-use-intent-v1",
            "one_use": True, "attempt_number": 1, "automatic_retries": 0,
            "identity": partition_identity(plan), "execution_platform": "databricks",
            "world_size": 1, "gpu_count": 1, "max_concurrent_runs": 1,
            "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
            "source_commit": "a" * 40, "source_sha256": "b" * 64,
            "timeout_seconds": timeout_seconds, "max_gpu_hours": max_gpu_hours,
            "wall_clock_stop_seconds": stop_seconds,
            "wall_clock_ceiling_seconds": timeout_seconds,
            "model_tree_sha256": "c" * 64, "model_receipt_sha256": "d" * 64,
            "dataset_tree_sha256": DATASET_TREE_SHA256,
            "silver_records_sha256": "e" * 64, "silver_record_count": 8,
            "semantics_records_sha256": "f" * 64, "semantics_record_count": 8,
            "global_journal_sha256": "1" * 64,
            "admission_data_binding_sha256": "2" * 64,
            "admission_exposure_binding_sha256": "3" * 64,
            "mlflow_experiment_path": "/Shared/research4_simulated_continuity",
            "plan": asdict(plan), "plan_sha256": plan.full_plan_sha256,
        }
        if reviewed_timeout_budget is not None:
            value["reviewed_timeout_budget"] = reviewed_timeout_budget
        value["intent_sha256"] = digest(value)
        return value

    def test_catalog_layout_is_exact_bronze_silver_gold_and_managed_volume(self):
        sql = "\n".join(catalog_ddl())
        self.assertEqual(CATALOG, "research4_simulated_continuity")
        self.assertEqual(VOLUME, "/Volumes/research4_simulated_continuity/bronze/model_data")
        self.assertEqual(DATASET_RELEASE_ROOT, VOLUME + "/dataset_releases/" +
                         DATASET_TREE_SHA256)
        for token in ("DESCRIBE CATALOG", ".bronze", ".silver", ".gold",
                      "DESCRIBE VOLUME", BRONZE_TABLE, SILVER_TABLE, GOLD_RUN_TABLE,
                      "dataset_release_path", "dataset_release_receipt_sha256"):
            self.assertIn(token, sql)

    def test_dataset_release_is_content_bound_before_table_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            for index in range(26):
                path = root / "data" / f"fixture-{index:02d}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"index": index}) + "\n")
                entries.append({"path": path.relative_to(root).as_posix(),
                                "sha256": file_sha256(path),
                                "bytes": path.stat().st_size})
            receipt = {"files": entries,
                       "dataset_tree_digest": DATASET_TREE_SHA256,
                       "family_split_digest": "a" * 64}
            receipt_path = root / "RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt))
            receipt_sha = file_sha256(receipt_path)
            validation_path = root / "VALIDATION.json"
            validation_path.write_text(json.dumps({"status": "valid"}))
            validation_sha = file_sha256(validation_path)
            proof = verify_dataset_release(
                root, expected_root=root,
                expected_receipt_sha256=receipt_sha,
                expected_validation_sha256=validation_sha)
            self.assertEqual(proof["checked_file_count"], 26)
            self.assertEqual(proof["dataset_tree_sha256"], DATASET_TREE_SHA256)
            entries[0]["path"] = "changed"
            (root / "data" / "fixture-00.json").write_text("tampered\n")
            with self.assertRaises(PermissionError):
                verify_dataset_release(root, expected_root=root,
                                       expected_receipt_sha256=receipt_sha,
                                       expected_validation_sha256=validation_sha)
        self.assertNotEqual(stable_rows_sha256(
            [{"record_id": "a"}], ("record_id",)),
            stable_rows_sha256([{"record_id": "b"}], ("record_id",)))

    def test_one_use_intent_and_shared_lock_fail_closed(self):
        intent = validate_intent(self.intent())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with VolumeOneUseLock(root, intent, owner="run-1"):
                with self.assertRaises((FileExistsError, PermissionError)):
                    with VolumeOneUseLock(root, self.intent(), owner="run-2"):
                        pass
            with self.assertRaises(PermissionError):
                with VolumeOneUseLock(root, intent, owner="run-3"):
                    pass
        changed = self.intent(); changed["plan"]["optimizer_updates"] = 3
        changed["intent_sha256"] = digest({k: v for k, v in changed.items()
                                            if k != "intent_sha256"})
        with self.assertRaises(PermissionError):
            validate_intent(changed)

    def test_owner_bound_closeout_releases_active_but_retains_consumed_identity(self):
        intent = validate_intent(self.intent())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reserve_intent(root, intent, "job-17")
            receipt = close_active_reservation(
                root, intent, "job-17", "partition_notebook_failed")
            self.assertFalse((root / "scientific.lock").exists())
            self.assertTrue((root / "consumed" / (intent["identity"] + ".json")).is_file())
            self.assertTrue(receipt["consumed_identity_retained"])
            self.assertEqual(receipt["receipt_sha256"], digest({
                key: value for key, value in receipt.items()
                if key != "receipt_sha256"}))
            self.assertEqual(
                close_active_reservation(root, intent, "job-17", "job_all_done_cleanup"),
                receipt)
            with self.assertRaises(PermissionError):
                close_active_reservation(root, intent, "job-18", "job_all_done_cleanup")
            with self.assertRaises(PermissionError):
                reserve_intent(root, intent, "job-19")

    def test_volume_snapshot_receipt_recomputes_shards_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"; root.mkdir()
            (root / "config.json").write_text("{}\n")
            shard = "model-00001-of-00001.safetensors"
            (root / shard).write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (root / "tokenizer.json").write_text("{}\n")
            (root / "tokenizer_config.json").write_text("{}\n")
            receipt = build_model_receipt(root)
            path = root.parent / "MODEL_SNAPSHOT_RECEIPT.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(verify_model_receipt(root, path)["resolved_commit"],
                             MODEL_REVISION)
            (root / shard).write_bytes(b"changed")
            with self.assertRaises(PermissionError):
                verify_model_receipt(root, path)

    def test_primary_authority_replays_raw_qualification_and_rejects_tamper(self):
        qualification_intent = validate_intent(self.intent())
        rank = {
            "status": "complete", "plan_sha256": qualification_intent["plan_sha256"],
            "successful_optimizer_updates": 2, "active_branch_changed": True,
            "tokens_seen": 4096, "elapsed_seconds": 16.0,
            "resources": {"gpu": {"nvml": {"nonzero_gpu_sample_count": 2}}},
        }
        rank["resources"]["tokens_per_second"] = 256.0
        rank["receipt_sha256"] = digest(rank)
        evidence = {
            "intent_sha256": qualification_intent["intent_sha256"],
            "rank_receipt_sha256": rank["receipt_sha256"],
            "live_base_components_before": {"weights": "a" * 64},
            "live_base_components_after": {"weights": "a" * 64},
            "save_reload_pass": True,
            "qualification_metrics": {"architecture_pass": True,
                                      "capacity_adequate": True},
        }
        evidence["evidence_sha256"] = digest(evidence)
        skill, _ = matrix_plans({
            "version": "qwen35-9b-ordinal-continuity-experiment-v1",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "conditions": ["self_continuity", "artifact_continuity", "label_only"],
            "seeds": [202610281, 202610282, 202610283],
            "stages": ["delta1", "delta2", "delta3"],
            "qualification_updates": 2, "skill_optimizer_updates": 24,
            "primary_optimizer_updates": 144, "skill_rank": 4,
            "skill_alpha": 8.0, "automatic_retries": 0, "one_use": True}, 4)
        skill = replace(skill, device_profile="databricks-a10x1", world_size=1,
                        micro_batch_size=2, accumulation_steps=2).validate()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root / "qualification_output"
            checkpoint = output / "checkpoint"; checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.safetensors").write_bytes(b"qualified")
            (output / "RANK_0_RECEIPT.json").write_text(json.dumps(rank))
            (output / "DATABRICKS_EVIDENCE.json").write_text(json.dumps(evidence))
            qualification_receipt = create_qualification_receipt(
                qualification_intent, rank, evidence,
                directory_tree_sha256(checkpoint))
            budget = primary_timeout_budget(skill, qualification_receipt)
            timeout = budget["derived_ceiling_seconds"]
            primary_intent = validate_intent(self.intent(
                plan=skill, timeout_seconds=timeout,
                max_gpu_hours=timeout / 3600,
                stop_seconds=timeout - 300,
                reviewed_timeout_budget=budget))
            qualification_intent_path = root / "qualification_intent.json"
            qualification_receipt_path = root / "qualification_receipt.json"
            admission_path = root / "primary_admission.json"
            qualification_intent_path.write_text(json.dumps(qualification_intent))
            qualification_receipt_path.write_text(json.dumps(qualification_receipt))
            admission_path.write_text(json.dumps(create_primary_admission(
                primary_intent, qualification_receipt)))
            replay = validate_primary_authority_files(
                primary_intent, admission_path, qualification_intent_path,
                output, qualification_receipt_path)
            self.assertEqual(replay["qualification_receipt"], qualification_receipt)
            invented = json.loads(json.dumps(primary_intent))
            invented_budget = invented["reviewed_timeout_budget"]
            invented_budget["qualification_tokens_per_second"] = 512.0
            invented_budget["derived_ceiling_seconds"] = 3984
            invented_budget["budget_sha256"] = digest({
                key: value for key, value in invented_budget.items()
                if key != "budget_sha256"})
            invented["timeout_seconds"] = invented["wall_clock_ceiling_seconds"] = 3984
            invented["wall_clock_stop_seconds"] = 3684
            invented["max_gpu_hours"] = 3984 / 3600
            invented["intent_sha256"] = digest({
                key: value for key, value in invented.items()
                if key != "intent_sha256"})
            validate_intent(invented)
            with self.assertRaises(PermissionError):
                create_primary_admission(invented, qualification_receipt)
            evidence["qualification_metrics"]["architecture_pass"] = False
            evidence["evidence_sha256"] = digest({
                key: value for key, value in evidence.items()
                if key != "evidence_sha256"})
            (output / "DATABRICKS_EVIDENCE.json").write_text(json.dumps(evidence))
            with self.assertRaises(PermissionError):
                validate_primary_authority_files(
                    primary_intent, admission_path, qualification_intent_path,
                    output, qualification_receipt_path)

    def test_gold_summary_requires_all_27_policy_partitions(self):
        skill = {"status": "complete", "optimizer_updates": 24, "stage": "skill",
                 "receipt_replayed": True, "receipt_sha256": "a" * 64,
                 "checkpoint_tree_sha256": "b" * 64,
                 "source_commit": "a" * 40, "model_tree_sha256": "c" * 64}
        spec = {
            "version": "qwen35-9b-ordinal-continuity-experiment-v1",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "conditions": ["self_continuity", "artifact_continuity", "label_only"],
            "seeds": [202610281, 202610282, 202610283],
            "stages": ["delta1", "delta2", "delta3"],
            "qualification_updates": 2, "skill_optimizer_updates": 24,
            "primary_optimizer_updates": 144, "skill_rank": 4,
            "skill_alpha": 8.0, "automatic_retries": 0, "one_use": True}
        skill_plan, matrix = matrix_plans(spec, 4)
        skill_plan = replace(skill_plan, device_profile="databricks-a10x1",
                             world_size=1, micro_batch_size=2,
                             accumulation_steps=2).validate()
        skill["plan_sha256"] = skill_plan.full_plan_sha256
        policies = []
        prior = {}
        index = 0
        plans = {(plan.condition, plan.seed, plan.stage): replace(
            plan, device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2).validate()
            for plan in matrix}
        for condition in ("self_continuity", "artifact_continuity", "label_only"):
            for seed in (202610281, 202610282, 202610283):
                parent = skill["checkpoint_tree_sha256"]
                for stage in ("delta1", "delta2", "delta3"):
                    checkpoint = f"{index + 10:064x}"
                    policies.append({"status": "complete", "optimizer_updates": 144,
                                     "receipt_replayed": True, "condition": condition,
                                     "seed": seed, "stage": stage,
                                     "parent_checkpoint_tree_sha256": parent,
                                     "checkpoint_tree_sha256": checkpoint,
                                     "receipt_sha256": f"{index:064x}",
                                     "source_commit": "a" * 40,
                                     "model_tree_sha256": "c" * 64,
                                     "plan_sha256": plans[(condition, seed, stage)].full_plan_sha256})
                    parent = checkpoint; index += 1
        self.assertEqual(gold_summary(skill, policies)["policy_partition_count"], 27)
        with self.assertRaises(PermissionError):
            gold_summary(skill, policies[:-1])

    def test_job_specs_are_bounded_offline_gpu_and_mlflow(self):
        root = Path(__file__).parents[1]
        job = (root / "resources/simulated_continuity_databricks.job.yml").read_text()
        source = (root / "src/databricks_continuity_partition.py").read_text()
        self.assertNotIn("kaggle", job.lower())
        bundle = (root / "databricks.yml").read_text()
        self.assertIn("default: GPU_1xA10", bundle)
        self.assertNotIn("Standard_NV", bundle)
        self.assertNotIn("timeout_seconds: 43200", job)
        self.assertGreaterEqual(job.count("max_retries: 0"), 8)
        self.assertIn("max_concurrent_runs: 1", job)
        jobs = yaml.safe_load(job)["resources"]["jobs"]
        for name in ("simulated_continuity_rank4_qualification",
                     "simulated_continuity_primary_partition"):
            spec = jobs[name]
            self.assertNotIn("job_clusters", spec)
            gpu = [task for task in spec["tasks"]
                   if task.get("compute") is not None]
            self.assertEqual(len(gpu), 1)
            self.assertEqual(gpu[0]["task_key"], "run_one_partition")
            self.assertEqual(gpu[0]["compute"], {
                "hardware_accelerator": "${var.continuity_gpu_accelerator}"})
            self.assertEqual(gpu[0]["environment_key"], "continuity_gpu")
            self.assertNotIn("libraries", gpu[0])
            environment = next(item for item in spec["environments"]
                               if item["environment_key"] == "continuity_gpu")
            self.assertEqual(environment["spec"]["environment_version"],
                             "${var.continuity_gpu_environment_version}")
            self.assertIn("torch==2.11.0+cu128",
                          environment["spec"]["dependencies"])
            self.assertIn("transformers==5.19.0",
                          environment["spec"]["dependencies"])
        self.assertEqual(jobs["simulated_continuity_rank4_qualification"]
                         ["timeout_seconds"], 4500)
        self.assertEqual(jobs["simulated_continuity_rank4_qualification"]
                         ["tasks"][1]["timeout_seconds"], 2700)
        self.assertEqual(jobs["simulated_continuity_primary_partition"]
                         ["timeout_seconds"], 16200)
        self.assertEqual(jobs["simulated_continuity_primary_partition"]
                         ["tasks"][1]["timeout_seconds"], 14400)
        self.assertIn('"HF_HUB_OFFLINE": "1"', source)
        self.assertIn('"TRANSFORMERS_OFFLINE": "1"', source)
        self.assertIn("mlflow.set_experiment", source)
        self.assertIn("verify_reserved_intent", source)
        self.assertIn('"src.databricks_continuity_partition_worker"', source)
        self.assertNotIn("${var.code_path}/research4", job)
        self.assertNotIn("active_lock.unlink", source)
        self.assertEqual(job.count("run_if: ALL_DONE"), 2)
        self.assertEqual(job.count("notebook_path: ../notebooks/databricks_continuity_cleanup.py"), 2)
        self.assertGreaterEqual(job.count("qualification_receipt_path"), 3)
        preflight = (root / "notebooks/databricks_continuity_preflight.py").read_text()
        self.assertIn("validate_primary_authority_files", preflight)
        self.assertIn("validate_exact_gold_row", preflight)
        data_notebook = (root / "notebooks/databricks_continuity_data.py").read_text()
        self.assertIn("verify_dataset_release(dataset_release_path)", data_notebook)
        self.assertIn("schema=spark.table(table).schema", data_notebook)
        self.assertNotIn('code_path + "/simulated_continuity_v1/generated"', data_notebook)
        self.assertIn("dataset_release_path: ${var.continuity_dataset_release}", job)
        self.assertNotIn("/local_disk0", source)
        model_stage = (root / "notebooks/databricks_continuity_model_snapshot.py").read_text()
        self.assertNotIn("/local_disk0", model_stage)

    def test_policy_predecessor_is_exact_canonical_partition(self):
        _, policies = matrix_plans({
            "version": "qwen35-9b-ordinal-continuity-experiment-v1",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "conditions": ["self_continuity", "artifact_continuity", "label_only"],
            "seeds": [202610281, 202610282, 202610283],
            "stages": ["delta1", "delta2", "delta3"],
            "qualification_updates": 2, "skill_optimizer_updates": 24,
            "primary_optimizer_updates": 144, "skill_rank": 4,
            "skill_alpha": 8.0, "automatic_retries": 0, "one_use": True}, 4)
        delta2 = next(value for value in policies
                      if value.condition == "self_continuity"
                      and value.seed == 202610281 and value.stage == "delta2")
        delta2 = replace(delta2, device_profile="databricks-a10x1", world_size=1,
                         micro_batch_size=2, accumulation_steps=2).validate()
        predecessor = expected_predecessor_binding(asdict(delta2))
        self.assertEqual(predecessor["stage"], "delta1")
        self.assertEqual(predecessor["condition"], "self_continuity")
        self.assertEqual(predecessor["seed"], 202610281)


if __name__ == "__main__":
    unittest.main()
