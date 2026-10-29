from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import yaml

from src.databricks_simulated_continuity import (
    BASE_QUALIFICATION_FAMILIES, BASE_QUALIFICATION_IDENTITY,
    BASE_QUALIFICATION_PROPOSAL_SHA256, BASE_QUALIFICATION_SEED,
    BASE_QUALIFICATION_SELECTION_SHA256, BASE_QUALIFICATION_SET_VERSION,
    BRONZE_TABLE, CATALOG, DATASET_RELEASE_ROOT,
    DATASET_RELEASE_RECEIPT_SHA256,
    DATASET_TREE_SHA256, EXPERIMENT_SPEC,
    GOLD_RUN_TABLE, INTENT_VERSION, MODEL_ID, MODEL_RECEIPT, MODEL_REVISION, MODEL_ROOT,
    SOURCE_MANIFEST_PATHS,
    SILVER_TABLE, VOLUME, VolumeOneUseLock,
    base_qualification_plan, build_base_qualification_intent, build_intent,
    build_model_receipt, build_source_manifest, canonical, catalog_ddl,
    close_active_reservation,
    commit_model_snapshot, create_base_qualification_receipt,
    create_primary_admission, create_qualification_receipt,
    digest, file_sha256,
    expected_predecessor_binding, gold_summary, primary_timeout_budget, reserve_intent,
    silver_partition_bindings, silver_rows, stable_rows_sha256,
    validate_base_qualification_silver_rows, validate_base_silver_binding,
    validate_base_qualification_intent, validate_base_qualification_receipt,
    validate_intent,
    validate_policy_silver_payload, validate_primary_authority_files,
    validate_source_manifest,
    verify_dataset_release, verify_deployed_source, verify_model_receipt,
    verify_model_receipt_authority)
from src.action_serialization import (
    SERIALIZATION_CONTRACT_VERSION, parse_registered_action,
    serialization_contract,
)
from src.databricks_continuity_partition import _stage_local_model
from src.qwen9b_ordinal_continuity_controller import matrix_plans, partition_identity
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qwen9b_ordinal_continuity_training import TrainingPlan
from src.qwen9b_ordinal_continuity_training import (
    adjacent_pair_schedule, canonical_sha256,
    load_simulator_condition, validate_matched_conditions)


class DatabricksContinuityTests(unittest.TestCase):
    def base_resources(self, model_tree_sha256="c" * 64):
        gib = 1024**3
        return {
            "gpu": {
                "profile": "databricks-a10x1", "compute_dtype": "float16",
                "adapter_dtype": "float32", "devices": [{
                    "index": 0, "name": "NVIDIA A10",
                    "total_memory_bytes": 24 * gib, "major": 8, "minor": 6,
                }],
            },
            "backend": {
                "model_class": "Qwen3_5ForConditionalGeneration",
                "linear_attention_layers": 24, "full_attention_layers": 8,
                "kernel_selection": {"profile": "fla"},
            },
            "model_copy": {
                "copy_seconds": 10.0, "copy_bytes": 10 * gib,
                "copy_bytes_per_second": gib,
                "local_disk_free_before": 24 * gib,
                "local_disk_free_after": 14 * gib,
                "tree_sha256": model_tree_sha256,
            },
            "local_disk_free_before_copy": 24 * gib,
            "elapsed_seconds": 120.0, "inference_seconds": 60.0,
            "prompt_tokens": 4096, "generated_tokens": 576,
            "batch_size": 16, "decisions_per_second": 4.8,
            "peak_gpu_memory_bytes": 12 * gib,
            "nvml": {"sample_count": 18, "mean_utilization_percent": 72.5,
                     "peak_memory_used_bytes": 13 * gib},
            "operator_stop_seconds": 2400, "ceiling_seconds": 2700,
            "protected_export_seconds": 300,
        }

    def base_binding(self, silver_records_sha256="e" * 64):
        value = {
            "version": "research4-databricks-silver-binding-v2",
            "partition_identity": BASE_QUALIFICATION_IDENTITY,
            "plan_sha256": base_qualification_plan()["plan_sha256"],
            "silver_records_sha256": silver_records_sha256,
            "silver_record_count": 288,
            "semantics_records_sha256": digest([]),
            "semantics_record_count": 0,
            "global_journal_sha256": silver_records_sha256,
            "data_binding_sha256": digest({
                "record_count": 288,
                "family_counts": {name: 36 for name in BASE_QUALIFICATION_FAMILIES},
                "training_eligible_count": 0,
                "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
                "qualification_seed": BASE_QUALIFICATION_SEED,
                "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
            }),
            "exposure_binding_sha256": digest({
                "frozen_base_only": True, "training_eligible": False,
                "optimizer_updates": 0, "records": 288,
                "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
                "qualification_seed": BASE_QUALIFICATION_SEED,
                "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
                "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
                "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION}),
        }
        value["binding_sha256"] = digest(value)
        return value

    def base_receipt(self):
        value = {
            "version": "research4-databricks-frozen-base-format-receipt-v2",
            "status": "qualified", "identity": BASE_QUALIFICATION_IDENTITY,
            "plan_sha256": base_qualification_plan()["plan_sha256"],
            "source_commit": "a" * 40, "source_sha256": "6" * 64,
            "model_tree_sha256": "c" * 64,
            "model_receipt_sha256": "d" * 64,
            "dataset_tree_sha256": DATASET_TREE_SHA256,
            "silver_records_sha256": "e" * 64,
            "output_root": (f"{VOLUME}/base_qualifications/{'a' * 40}/"
                            f"{BASE_QUALIFICATION_IDENTITY}"),
            "base_qualification_intent_sha256": "4" * 64,
            "adapter_parameter_count": 0, "optimizer_updates": 0,
            "backward_passes": 0, "decision_count": 288,
            "strict_schema_and_legal_count": 288,
            "target_match_count_secondary": 280,
            "family_strict_schema_and_legal": {
                name: 36 for name in BASE_QUALIFICATION_FAMILIES},
            "serialization_contract": serialization_contract(),
            "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
            "normalization_used_count": 0,
            "family_normalization_used": {
                name: 0 for name in BASE_QUALIFICATION_FAMILIES},
            "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
            "qualification_seed": BASE_QUALIFICATION_SEED,
            "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
            "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
            "minimum_required": 286, "minimum_required_per_family": 35,
            "base_component_sha256_before": "7" * 64,
            "base_component_sha256_after": "7" * 64,
            "raw_results_sha256": "8" * 64,
            "resources": self.base_resources(),
        }
        value["receipt_sha256"] = digest(value)
        return value

    def intent(self, *, plan=None, timeout_seconds=2700, max_gpu_hours=0.75,
               stop_seconds=2400, reviewed_timeout_budget=None):
        plan = (plan or TrainingPlan(
            device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2)).validate()
        source_manifest = build_source_manifest(
            Path(__file__).parents[1], "a" * 40)
        value = {
            "version": INTENT_VERSION,
            "one_use": True, "attempt_number": 1, "automatic_retries": 0,
            "identity": partition_identity(plan), "execution_platform": "databricks",
            "world_size": 1, "gpu_count": 1, "max_concurrent_runs": 1,
            "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
            "source_commit": "a" * 40, "source_manifest": source_manifest,
            "source_sha256": source_manifest["manifest_sha256"],
            "timeout_seconds": timeout_seconds, "max_gpu_hours": max_gpu_hours,
            "wall_clock_stop_seconds": stop_seconds,
            "wall_clock_ceiling_seconds": timeout_seconds,
            "model_tree_sha256": "c" * 64, "model_receipt_sha256": "d" * 64,
            "dataset_tree_sha256": DATASET_TREE_SHA256,
            "silver_records_sha256": "e" * 64, "silver_record_count": 2304,
            "semantics_records_sha256": digest([]), "semantics_record_count": 0,
            "global_journal_sha256": "1" * 64,
            "admission_data_binding_sha256": "2" * 64,
            "admission_exposure_binding_sha256": "3" * 64,
            "base_qualification_intent_sha256": "4" * 64,
            "base_qualification_receipt_sha256": "5" * 64,
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

    def test_source_manifest_normalizes_newlines_and_replays_deployed_notebooks(self):
        source = Path(__file__).parents[1]
        commit = "a" * 40
        expected = build_source_manifest(source, commit)
        self.assertEqual(validate_source_manifest(expected), expected)
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "canonical"
            deployed = Path(folder) / "research4" / "source_releases" / commit
            for logical in SOURCE_MANIFEST_PATHS:
                raw = (source / logical).read_bytes().decode("utf-8")
                canonical = raw.replace("\r\n", "\n").replace("\r", "\n")
                local_target = copied / logical
                local_target.parent.mkdir(parents=True, exist_ok=True)
                local_target.write_text(canonical.replace("\n", "\r\n"), newline="")
                deployed_logical = (Path(logical).with_suffix("")
                                    if logical.startswith("notebooks/") else Path(logical))
                deployed_target = deployed / deployed_logical
                deployed_target.parent.mkdir(parents=True, exist_ok=True)
                deployed_target.write_text(canonical, newline="")
            self.assertEqual(build_source_manifest(copied, commit), expected)
            intent = {"source_commit": commit, "source_manifest": expected,
                      "source_sha256": expected["manifest_sha256"]}
            self.assertEqual(verify_deployed_source(deployed, intent), expected)
            target = deployed / "src" / "composite_kernels.py"
            target.write_text(target.read_text() + "# tamper\n")
            with self.assertRaises(PermissionError):
                verify_deployed_source(deployed, intent)
            ambiguous = copied / "notebooks" / "databricks_continuity_preflight"
            ambiguous.write_text((copied / "notebooks" /
                                  "databricks_continuity_preflight.py").read_text())
            with self.assertRaises(PermissionError):
                build_source_manifest(copied, commit)

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

    def test_frozen_base_format_gate_is_zero_update_and_thresholded(self):
        source = Path(__file__).parents[1]
        manifest = build_source_manifest(source, "a" * 40)
        outcome_binding = sorted(({
            "record_id": f"f{family_index}-{index}",
            "record_sha256": hashlib.sha256(
                f"f{family_index}-{index}".encode()).hexdigest(),
        } for family_index, _ in enumerate(BASE_QUALIFICATION_FAMILIES)
            for index in range(36)),
            key=lambda row: row["record_id"])
        binding = self.base_binding(digest(outcome_binding))
        intent = build_base_qualification_intent(
            source_commit="a" * 40, source_manifest=manifest,
            model_tree_sha256="c" * 64,
            model_receipt_sha256="d" * 64,
            silver_binding=binding,
            mlflow_experiment_path="/Shared/research4_simulated_continuity")
        self.assertEqual(validate_base_qualification_intent(intent), intent)
        self.assertEqual(intent["plan"]["optimizer_updates"], 0)
        self.assertFalse(intent["plan"]["adapter_contribution"])
        outcomes = []
        for family_index, family in enumerate(BASE_QUALIFICATION_FAMILIES):
            for index in range(36):
                text = ('```json\n{"action":"N01"}\n```'
                        if family_index == 0 and index == 0
                        else '{"action":"N01"}')
                target = "N01" if index % 2 == 0 else "N02"
                parsed = parse_registered_action(text, ["N01", "N02"])
                outcomes.append({
                    "record_id": f"f{family_index}-{index}",
                    "record_sha256": hashlib.sha256(
                        f"f{family_index}-{index}".encode()).hexdigest(),
                    "family_id": family,
                    "strict_json": parsed["valid"],
                    "serialization_valid": parsed["valid"],
                    "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
                    "normalization_used": parsed["normalization_used"],
                    "serialization_form": parsed["serialization_form"],
                    "raw_response_sha256": parsed["raw_response_sha256"],
                    "legal_action": parsed["valid"],
                    "target_match": parsed["valid"] and parsed["action"] == target,
                    "action": parsed["action"],
                    "allowed_actions": ["N01", "N02"],
                    "target_action_id": target, "output_text": text,
                    "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                })
        receipt = create_base_qualification_receipt(
            intent, outcomes, raw_results_sha256="8" * 64,
            base_component_sha256_before="7" * 64,
            base_component_sha256_after="7" * 64,
            resource_metrics=self.base_resources())
        self.assertEqual(validate_base_qualification_receipt(receipt), receipt)
        self.assertEqual(receipt["strict_schema_and_legal_count"], 288)
        self.assertEqual(receipt["normalization_used_count"], 1)
        self.assertEqual(receipt["family_normalization_used"][
            BASE_QUALIFICATION_FAMILIES[0]], 1)
        changed_resource = json.loads(json.dumps(receipt))
        changed_resource["resources"]["gpu"]["profile"] = "unreviewed-gpu"
        changed_resource["receipt_sha256"] = digest({
            key: value for key, value in changed_resource.items()
            if key != "receipt_sha256"})
        with self.assertRaises(PermissionError):
            validate_base_qualification_receipt(changed_resource)
        changed_silver = json.loads(json.dumps(outcomes))
        changed_silver[0]["record_sha256"] = "f" * 64
        with self.assertRaises(PermissionError):
            create_base_qualification_receipt(
                intent, changed_silver, raw_results_sha256="8" * 64,
                base_component_sha256_before="7" * 64,
                base_component_sha256_after="7" * 64,
                resource_metrics=self.base_resources())
        invented_binding = dict(binding)
        invented_binding["global_journal_sha256"] = "1" * 64
        invented_binding["binding_sha256"] = digest({
            key: value for key, value in invented_binding.items()
            if key != "binding_sha256"})
        with self.assertRaises(PermissionError):
            build_base_qualification_intent(
                source_commit="a" * 40, source_manifest=manifest,
                model_tree_sha256="c" * 64,
                model_receipt_sha256="d" * 64,
                silver_binding=invented_binding,
                mlflow_experiment_path="/Shared/research4_simulated_continuity")
        forged = json.loads(json.dumps(outcomes))
        forged[0]["legal_action"] = False
        with self.assertRaises(PermissionError):
            create_base_qualification_receipt(
                intent, forged, raw_results_sha256="8" * 64,
                base_component_sha256_before="7" * 64,
                base_component_sha256_after="7" * 64,
                resource_metrics=self.base_resources())
        for index in (0, 1):
            outcomes[index].update({
                "strict_json": False, "serialization_valid": False,
                "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
                "normalization_used": False,
                "serialization_form": "bare_json_object",
                "raw_response_sha256": hashlib.sha256(b"not-json").hexdigest(),
                "legal_action": False, "target_match": False, "action": None,
                "output_text": "not-json",
                "output_sha256": hashlib.sha256(b"not-json").hexdigest(),
            })
        failed = create_base_qualification_receipt(
            intent, outcomes, raw_results_sha256="8" * 64,
            base_component_sha256_before="7" * 64,
            base_component_sha256_after="7" * 64,
            resource_metrics=self.base_resources())
        self.assertEqual(failed["status"], "failed")
        with self.assertRaises(PermissionError):
            validate_base_qualification_receipt(failed)

    def test_silver_bindings_replay_raw_schema_but_hash_tokenized_rows(self):
        class ToyTokenizer:
            eos_token_id = 9
            def apply_chat_template(self, messages, tokenize,
                                    add_generation_prompt, enable_thinking):
                return [1, 2, 3, 4]
            def encode(self, value, add_special_tokens=False):
                return [8] if value == " " else [5, 6, 7]

        generated = Path(__file__).parents[1] / "simulated_continuity_v1/generated"
        silver = silver_rows(generated, "a" * 40, tokenizer=ToyTokenizer())
        self.assertEqual({row["split"] for row in silver}, {"train", "qualification"})
        self.assertEqual(sum(row["admitted_for_training"] for row in silver), 6912)
        self.assertEqual(sum(not row["admitted_for_training"] for row in silver), 288)
        self.assertTrue(all(row["record_sha256"] == hashlib.sha256(
            row["record_json"].encode()).hexdigest() for row in silver))
        bindings = silver_partition_bindings(silver, generated)
        self.assertEqual(len(bindings), 29)
        qualification = bindings["qualification_rank4"]
        self.assertEqual(qualification["silver_record_count"], 2304)
        self.assertEqual(qualification["semantics_record_count"], 0)
        self.assertEqual(qualification["binding_sha256"], digest({
            key: value for key, value in qualification.items()
            if key != "binding_sha256"}))
        raw_conditions = {condition: load_simulator_condition(generated, condition)
                          for condition in EXPERIMENT_SPEC["conditions"]}
        self.assertEqual(qualification["data_binding_sha256"], canonical_sha256(
            validate_matched_conditions(raw_conditions)))
        base = bindings[BASE_QUALIFICATION_IDENTITY]
        self.assertEqual(validate_base_silver_binding(base), base)
        self.assertEqual(base["silver_record_count"], 288)
        self.assertEqual(base["semantics_record_count"], 0)
        self.assertEqual(base["plan_sha256"], base_qualification_plan()["plan_sha256"])
        base_intent = build_base_qualification_intent(
            source_commit="a" * 40,
            source_manifest=build_source_manifest(Path(__file__).parents[1], "a" * 40),
            model_tree_sha256="c" * 64,
            model_receipt_sha256="d" * 64,
            silver_binding=base,
            mlflow_experiment_path="/Shared/research4_simulated_continuity")
        base_rows = []
        for row in silver:
            if row["record_kind"] != "base_qualification":
                continue
            item = dict(row)
            item.update({
                "dataset_release_path": DATASET_RELEASE_ROOT,
                "dataset_release_receipt_sha256": DATASET_RELEASE_RECEIPT_SHA256,
            })
            base_rows.append(item)
        validated_base = validate_base_qualification_silver_rows(
            base_rows, base_intent)
        self.assertEqual(len(validated_base), 288)
        changed_payload = [dict(row) for row in base_rows]
        changed_payload[0]["record_json"] += " "
        with self.assertRaises(PermissionError):
            validate_base_qualification_silver_rows(changed_payload, base_intent)
        changed_tree = [dict(row) for row in base_rows]
        changed_tree[0]["dataset_tree_sha256"] = "f" * 64
        with self.assertRaises(PermissionError):
            validate_base_qualification_silver_rows(changed_tree, base_intent)
        invented_binding = dict(base)
        invented_binding["partition_identity"] = "invented"
        invented_binding["binding_sha256"] = digest({
            key: value for key, value in invented_binding.items()
            if key != "binding_sha256"})
        with self.assertRaises(PermissionError):
            validate_base_silver_binding(invented_binding)
        qualification_plan = TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2,
            device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2).validate()
        tokenized = [json.loads(row["record_json"]) for row in silver
                     if row["record_kind"] == "self_continuity"]
        self.assertTrue(tokenized)
        self.assertEqual({row["condition"] for row in tokenized},
                         {"self_continuity"})
        validated = validate_policy_silver_payload(
            canonical(tokenized[0]), record_id=tokenized[0]["record_id"],
            expected_condition="self_continuity",
            expected_family_id=tokenized[0]["family_id"],
            expected_danger_level=tokenized[0]["danger_level"])
        self.assertEqual(validated, tokenized[0])
        for field, mutation in (("condition", None),
                                ("condition", "artifact_continuity"),
                                ("record_kind", "invented"),
                                ("serialization_contract_version", "old")):
            changed = dict(tokenized[0])
            if mutation is None:
                changed.pop(field)
            else:
                changed[field] = mutation
            with self.subTest(field=field, mutation=mutation), \
                    self.assertRaises(PermissionError):
                validate_policy_silver_payload(
                    canonical(changed), record_id=tokenized[0]["record_id"],
                    expected_condition="self_continuity",
                    expected_family_id=tokenized[0]["family_id"],
                    expected_danger_level=tokenized[0]["danger_level"])
        for family, danger in ((None, tokenized[0]["danger_level"]),
                               (tokenized[0]["family_id"], None),
                               ("wrong-family", tokenized[0]["danger_level"]),
                               (tokenized[0]["family_id"],
                                (tokenized[0]["danger_level"] + 1) % 4)):
            with self.subTest(family=family, danger=danger), \
                    self.assertRaises(PermissionError):
                validate_policy_silver_payload(
                    canonical(tokenized[0]),
                    record_id=tokenized[0]["record_id"],
                    expected_condition="self_continuity",
                    expected_family_id=family,
                    expected_danger_level=danger)
        outer_changed = [dict(row) for row in silver]
        target = next(row for row in outer_changed
                      if row["record_kind"] == "self_continuity")
        target["condition"] = "artifact_continuity"
        with self.assertRaises(PermissionError):
            silver_partition_bindings(outer_changed, generated)
        self.assertEqual(qualification["global_journal_sha256"],
                         canonical_sha256(adjacent_pair_schedule(
                             tokenized, qualification_plan)))
        corrupted = [dict(row) for row in silver]
        corrupted[0]["record_json"] = corrupted[0]["record_json"] + " "
        with self.assertRaises(PermissionError):
            silver_partition_bindings(corrupted, generated)
        with self.assertRaises((FileNotFoundError, PermissionError)):
            silver_partition_bindings(silver, generated / "missing")

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
            with self.assertRaises(ValueError):
                reserve_intent(root, intent, "")
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

    def test_model_receipt_full_verification_inventories_payload_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"; root.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (root / "config.json").write_text("{}\n")
            (root / shard).write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (root / "tokenizer.json").write_text("{}\n")
            (root / "tokenizer_config.json").write_text("{}\n")
            receipt = build_model_receipt(root)
            path = root.parent / "MODEL_SNAPSHOT_RECEIPT.json"
            path.write_text(json.dumps(receipt))
            from src import databricks_simulated_continuity as contract
            original = contract.snapshot_inventory
            with patch.object(contract, "snapshot_inventory",
                              wraps=original) as inventory:
                self.assertEqual(verify_model_receipt(root, path), receipt)
            self.assertEqual(inventory.call_count, 1)

    def test_lightweight_model_receipt_authority_rejects_descriptor_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"; root.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (root / "config.json").write_text("{}\n")
            (root / shard).write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (root / "tokenizer.json").write_text("{}\n")
            (root / "tokenizer_config.json").write_text("{}\n")
            receipt = build_model_receipt(root)
            path = root.parent / "MODEL_SNAPSHOT_RECEIPT.json"
            path.write_text(json.dumps(receipt))
            from src import databricks_simulated_continuity as contract
            with patch.object(contract, "snapshot_inventory",
                              side_effect=AssertionError("payload was hashed")):
                self.assertEqual(verify_model_receipt_authority(
                    path, expected_tree_sha256=receipt["tree_sha256"],
                    expected_receipt_sha256=receipt["receipt_sha256"]), receipt)
            tampered = json.loads(json.dumps(receipt))
            tampered["weight_map_sha256"] = "f" * 64
            tampered["receipt_sha256"] = digest({
                key: value for key, value in tampered.items()
                if key != "receipt_sha256"})
            path.write_text(json.dumps(tampered))
            with self.assertRaises(PermissionError):
                verify_model_receipt_authority(
                    path, expected_tree_sha256=receipt["tree_sha256"],
                    expected_receipt_sha256=receipt["receipt_sha256"])

    def test_volume_snapshot_commit_copies_verifies_and_publishes_receipt_last(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder); staged = parent / "staged"; staged.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (staged / "config.json").write_text("{}\n")
            (staged / shard).write_bytes(b"weights")
            (staged / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (staged / "tokenizer.json").write_text("{}\n")
            (staged / "tokenizer_config.json").write_text("{}\n")
            root, receipt = parent / "model", parent / "MODEL_SNAPSHOT_RECEIPT.json"
            proof = commit_model_snapshot(staged, root, receipt)
            self.assertEqual(proof, verify_model_receipt(root, receipt))
            self.assertEqual((root / shard).read_bytes(), b"weights")
            with self.assertRaises(FileExistsError):
                commit_model_snapshot(staged, root, receipt)

        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder); staged = parent / "staged"; staged.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (staged / "config.json").write_text("{}\n")
            (staged / shard).write_bytes(b"weights")
            (staged / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (staged / "tokenizer.json").write_text("{}\n")
            (staged / "tokenizer_config.json").write_text("{}\n")
            root, receipt = parent / "model", parent / "MODEL_SNAPSHOT_RECEIPT.json"
            with patch("src.databricks_simulated_continuity.shutil.copytree",
                       side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    commit_model_snapshot(staged, root, receipt)
            self.assertFalse(root.exists())
            self.assertFalse(receipt.exists())

    def test_local_model_staging_cleans_copy_on_expected_tree_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder); source = parent / "source"; source.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (source / "config.json").write_text("{}\n")
            (source / shard).write_bytes(b"weights")
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (source / "tokenizer.json").write_text("{}\n")
            (source / "tokenizer_config.json").write_text("{}\n")
            receipt = parent / "MODEL_SNAPSHOT_RECEIPT.json"
            authority = build_model_receipt(source)
            receipt.write_text(json.dumps(authority))
            with patch("src.databricks_continuity_partition.tempfile.gettempdir",
                       return_value=folder):
                with self.assertRaises(PermissionError):
                    _stage_local_model(
                        source, receipt, "tree-mismatch-test", "f" * 64,
                        authority["receipt_sha256"])
            self.assertFalse(
                (parent / "research4_model_cache" / "tree-mismatch-test").exists())

    def test_model_verification_order_is_cpu_full_gpu_light_copy_full(self):
        root = Path(__file__).parents[1]
        base_cpu = (root / "notebooks" /
                    "databricks_continuity_base_qualification_preflight.py").read_text()
        training_cpu = (root / "notebooks" /
                        "databricks_continuity_preflight.py").read_text()
        base_gpu = (root / "notebooks" /
                    "databricks_continuity_base_qualification.py").read_text()
        training_gpu = (root / "notebooks" /
                        "databricks_continuity_partition.py").read_text()
        for source in (base_cpu, training_cpu):
            self.assertIn("verify_model_receipt(", source)
            self.assertNotIn("verify_model_receipt_authority(", source)
        for source in (base_gpu, training_gpu):
            self.assertIn("verify_model_receipt_authority(", source)
            self.assertNotIn("verify_model_receipt(", source)

        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder); source = parent / "source"; source.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (source / "config.json").write_text("{}\n")
            (source / shard).write_bytes(b"weights")
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.weight": shard}}))
            (source / "tokenizer.json").write_text("{}\n")
            (source / "tokenizer_config.json").write_text("{}\n")
            authority = build_model_receipt(source)
            receipt = parent / "MODEL_SNAPSHOT_RECEIPT.json"
            receipt.write_text(json.dumps(authority))
            from src import databricks_continuity_partition as launcher
            light = launcher.verify_model_receipt_authority
            full = launcher.verify_model_receipt
            order = []
            def record_light(*args, **kwargs):
                order.append("gpu_light_authority")
                return light(*args, **kwargs)
            def record_full(*args, **kwargs):
                order.append("post_copy_full_hash")
                return full(*args, **kwargs)
            with patch.object(launcher.tempfile, "gettempdir", return_value=folder), \
                    patch.object(launcher, "verify_model_receipt_authority",
                                 side_effect=record_light), \
                    patch.object(launcher, "verify_model_receipt",
                                 side_effect=record_full):
                local_model, _, _ = _stage_local_model(
                    source, receipt, "verification-order-test",
                    authority["tree_sha256"], authority["receipt_sha256"])
            self.assertEqual(order, ["gpu_light_authority", "post_copy_full_hash"])
            shutil.rmtree(local_model.parent)

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
        primary = next(plan for plan in matrix_plans(EXPERIMENT_SPEC, 4)
                       if plan.condition == "self_continuity"
                       and plan.seed == 202610281 and plan.stage == "delta1")
        primary = replace(primary, device_profile="databricks-a10x1", world_size=1,
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
            budget = primary_timeout_budget(primary, qualification_receipt)
            timeout = budget["derived_ceiling_seconds"]
            primary_intent = validate_intent(self.intent(
                plan=primary, timeout_seconds=timeout,
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
            with self.assertRaises(PermissionError):
                validate_intent(invented)
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
        base = self.base_receipt()
        capacity = {
            "status": "qualified", "source_commit": "a" * 40,
            "model_tree_sha256": "c" * 64,
            "selected_rank": 4,
            "authorized_primary_plan_sha256s": sorted(
                replace(plan, device_profile="databricks-a10x1", world_size=1,
                        micro_batch_size=2, accumulation_steps=2).validate()
                .full_plan_sha256 for plan in matrix_plans(EXPERIMENT_SPEC, 4)),
            "base_qualification_receipt_sha256": base["receipt_sha256"],
        }
        capacity["receipt_sha256"] = digest(capacity)
        capacity_row = {
            "status": "complete", "mode": "qualification",
            "optimizer_updates": 2, "stage": "delta1",
            "parent_checkpoint_tree_sha256": None,
            "receipt_replayed": True,
            "qualification_receipt_sha256": base["receipt_sha256"],
            "receipt_sha256": "9" * 64,
            "source_commit": "a" * 40, "model_tree_sha256": "c" * 64,
        }
        matrix = matrix_plans(EXPERIMENT_SPEC, 4)
        policies = []
        prior = {}
        index = 0
        plans = {(plan.condition, plan.seed, plan.stage): replace(
            plan, device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2).validate()
            for plan in matrix}
        for condition in ("self_continuity", "artifact_continuity", "label_only"):
            for seed in (202610281, 202610282, 202610283):
                parent = None
                for stage in ("delta1", "delta2", "delta3"):
                    checkpoint = f"{index + 10:064x}"
                    policies.append({"status": "complete", "optimizer_updates": 144,
                                     "receipt_replayed": True, "condition": condition,
                                     "seed": seed, "stage": stage,
                                     "parent_checkpoint_tree_sha256": parent,
                                     "checkpoint_tree_sha256": checkpoint,
                                     "receipt_sha256": f"{index:064x}",
                                     "qualification_receipt_sha256": capacity[
                                         "receipt_sha256"],
                                     "source_commit": "a" * 40,
                                     "model_tree_sha256": "c" * 64,
                                     "plan_sha256": plans[(condition, seed, stage)].full_plan_sha256})
                    parent = checkpoint; index += 1
        self.assertEqual(gold_summary(
            base, capacity, capacity_row, policies)["policy_partition_count"], 27)
        with self.assertRaises(PermissionError):
            gold_summary(base, capacity, capacity_row, policies[:-1])

    def test_job_specs_are_bounded_offline_gpu_and_mlflow(self):
        root = Path(__file__).parents[1]
        job = (root / "resources/simulated_continuity_databricks.job.yml").read_text()
        source = (root / "src/databricks_continuity_partition.py").read_text()
        base_notebook = (root / "notebooks/databricks_continuity_base_qualification.py").read_text()
        base_preflight = (root / "notebooks/databricks_continuity_base_qualification_preflight.py").read_text()
        self.assertNotIn("kaggle", job.lower())
        bundle = (root / "databricks.yml").read_text()
        self.assertIn("default: GPU_1xA10", bundle)
        self.assertNotIn("Standard_NV", bundle)
        self.assertNotIn("timeout_seconds: 43200", job)
        self.assertGreaterEqual(job.count("max_retries: 0"), 8)
        self.assertIn("max_concurrent_runs: 1", job)
        jobs = yaml.safe_load(job)["resources"]["jobs"]
        expected_gpu_dependencies = [
            "--extra-index-url https://download.pytorch.org/whl/cu128",
            "torch==2.11.0+cu128", "transformers==5.19.0",
            "bitsandbytes==0.49.2", "accelerate==1.15.0",
            "safetensors==0.8.0", "fla-core==0.5.2", "einops==0.8.2",
            "numpy==2.2.2", "psutil==7.2.2", "triton==3.6.0",
            "nvidia-ml-py==13.615.71", "mlflow==3.1.0",
        ]
        for name in ("simulated_continuity_frozen_base_qualification",
                     "simulated_continuity_rank4_qualification",
                     "simulated_continuity_primary_partition"):
            spec = jobs[name]
            self.assertNotIn("job_clusters", spec)
            gpu = [task for task in spec["tasks"]
                   if task.get("compute") is not None]
            self.assertEqual(len(gpu), 1)
            self.assertEqual(gpu[0]["task_key"], (
                "run_frozen_base_qualification" if "frozen_base" in name
                else "run_one_partition"))
            self.assertEqual(gpu[0]["compute"], {
                "hardware_accelerator": "${var.continuity_gpu_accelerator}"})
            self.assertEqual(gpu[0]["environment_key"], "continuity_gpu")
            self.assertNotIn("libraries", gpu[0])
            environment = next(item for item in spec["environments"]
                               if item["environment_key"] == "continuity_gpu")
            self.assertEqual(environment["spec"]["environment_version"],
                             "${var.continuity_gpu_environment_version}")
            self.assertEqual(environment["spec"]["dependencies"],
                             expected_gpu_dependencies)
            self.assertEqual(sum(
                task.get("run_if") == "ALL_DONE" for task in spec["tasks"]), 1)
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
        self.assertIn("validate_policy_silver_payload", source)
        self.assertIn('"src.databricks_continuity_partition_worker"', source)
        self.assertIn('hashlib.sha256(row.record_json.encode("utf-8"))', source)
        self.assertIn('remaining_stop = intent["wall_clock_stop_seconds"] - prework_seconds',
                      source)
        self.assertLess(source.index('saveAsTable(GOLD_RUN_TABLE)'),
                        source.index('mlflow.end_run(status="FINISHED")'))
        self.assertLess(source.index('mlflow.end_run(status="FINISHED")'),
                        source.index("SET status = 'complete'"))
        self.assertNotIn("${var.code_path}/research4", job)
        self.assertNotIn("active_lock.unlink", source)
        self.assertEqual(job.count("run_if: ALL_DONE"), 4)
        for notebook in (base_preflight, base_notebook):
            self.assertLess(notebook.index("bootstrap_source_sha256 ="),
                            notebook.index("sys.path.insert"))
            self.assertIn('"record_json"', notebook)
            self.assertIn("validate_base_qualification_silver_rows", notebook)
        self.assertLess(base_notebook.index("task_started = time.monotonic()"),
                        base_notebook.index("verify_model_receipt"))
        self.assertLess(base_notebook.index('saveAsTable(GOLD_RUN_TABLE)'),
                        base_notebook.index('mlflow.end_run(status="FINISHED"'))
        self.assertLess(base_notebook.index('mlflow.end_run(status="FINISHED"'),
                        base_notebook.index("SET status = '{target_gold_status}'"))
        self.assertEqual(job.count("notebook_path: ../notebooks/databricks_continuity_cleanup.py"), 4)
        cleanup = (root / "notebooks/databricks_continuity_cleanup.py").read_text()
        self.assertNotIn("from src.", cleanup)
        self.assertIn("Independent ALL_DONE release", cleanup)
        self.assertIn("consumed_identity_retained", cleanup)
        self.assertEqual(job.count("job_start_timestamp_ms: '{{job.start_time.timestamp_ms}}'"), 4)
        self.assertGreaterEqual(job.count("qualification_receipt_path"), 3)
        preflight = (root / "notebooks/databricks_continuity_preflight.py").read_text()
        self.assertIn("validate_primary_authority_files", preflight)
        self.assertIn("validate_exact_gold_row", preflight)
        self.assertIn("verify_deployed_source", preflight)
        self.assertIn("validate_policy_silver_payload", preflight)
        self.assertIn('hashlib.sha256(row.record_json.encode("utf-8"))', preflight)
        self.assertNotIn("reserve_intent(", preflight)
        gpu_notebook = (root / "notebooks/databricks_continuity_partition.py").read_text()
        self.assertNotIn("%uv pip install", gpu_notebook)
        for pin in ("torch==2.11.0+cu128", "transformers==5.19.0",
                    "bitsandbytes==0.49.2", "nvidia-ml-py==13.615.71"):
            self.assertIn(pin, expected_gpu_dependencies)
        base_notebook = (root / "notebooks" /
                         "databricks_continuity_base_qualification.py").read_text()
        self.assertIn("mean_gpu_utilization_percent", base_notebook)
        self.assertIn("batch_size = 16", base_notebook)
        self.assertLess(gpu_notebook.index("verify_deployed_source("),
                        gpu_notebook.index("reserve_intent("))
        self.assertLess(gpu_notebook.index("bitsandbytes.functional.quantize_4bit("),
                        gpu_notebook.index("reserve_intent("))
        self.assertLess(gpu_notebook.index("shutil.disk_usage("),
                        gpu_notebook.index("reserve_intent("))
        self.assertLess(gpu_notebook.index("reserve_intent("),
                        gpu_notebook.index("result = run_partition("))
        data_notebook = (root / "notebooks/databricks_continuity_data.py").read_text()
        self.assertIn("verify_dataset_release(dataset_release_path)", data_notebook)
        self.assertIn("silver_partition_bindings(silver, generated)", data_notebook)
        self.assertIn("schema=spark.table(table).schema", data_notebook)
        self.assertNotIn('code_path + "/simulated_continuity_v1/generated"', data_notebook)
        self.assertIn("dataset_release_path: ${var.continuity_dataset_release}", job)
        self.assertNotIn("/local_disk0", source)
        model_stage = (root / "notebooks/databricks_continuity_model_snapshot.py").read_text()
        self.assertNotIn("/local_disk0", model_stage)
        self.assertIn("commit_model_snapshot(staged, root, receipt)", model_stage)
        self.assertNotIn("os.replace(staged, root)", model_stage)
        intent_tool = (root / "tools/prepare_databricks_continuity_intent.py").read_text()
        self.assertNotIn('"source-sha256"', intent_tool)
        self.assertIn("build_source_manifest", intent_tool)
        self.assertLess(source.index("verify_reserved_intent(lock_root"),
                        source.index(
                            "model_receipt = verify_model_receipt_authority("))
        self.assertLess(source.index("verify_reserved_intent("),
                        source.index("local_model, local_model_receipt, cache_metrics = "
                                     "_stage_local_model("))

    def test_policy_predecessor_is_exact_canonical_partition(self):
        policies = matrix_plans(EXPERIMENT_SPEC, 4)
        delta2 = next(value for value in policies
                      if value.condition == "self_continuity"
                      and value.seed == 202610281 and value.stage == "delta2")
        delta2 = replace(delta2, device_profile="databricks-a10x1", world_size=1,
                         micro_batch_size=2, accumulation_steps=2).validate()
        predecessor = expected_predecessor_binding(asdict(delta2))
        self.assertEqual(predecessor["stage"], "delta1")
        self.assertEqual(predecessor["condition"], "self_continuity")
        self.assertEqual(predecessor["seed"], 202610281)
        delta1 = next(value for value in policies
                      if value.condition == "self_continuity"
                      and value.seed == 202610281 and value.stage == "delta1")
        delta1 = replace(delta1, device_profile="databricks-a10x1", world_size=1,
                         micro_batch_size=2, accumulation_steps=2).validate()
        self.assertIsNone(expected_predecessor_binding(asdict(delta1)))


if __name__ == "__main__":
    unittest.main()
