import copy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.databricks_continuity_partition import (
    _EXPECTED_MLFLOW_ARTIFACTS,
    _ensure_pending_gold_row, _reconcile_mlflow_scalars,
    _publish_parent_evidence, _replay_completed_child, run_partition,
    _V7_RECOVERY_ORIGINAL, PARENT_RECOVERY_AUTHORITY_VERSION,
    _validate_parent_finalization,
    publish_parent_recovery_result,
    recover_parent_evidence,
    validate_parent_recovery_authority,
    _verify_completed_mlflow_state, _verify_finalizer_owner,
)
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, GOLD_RUN_TABLE, digest)
from src.databricks_continuity_partition_worker import (
    MLFLOW_JOURNAL_VERSION, _digest, run as run_partition_worker,
    validate_mlflow_journal,
)
from src.volume_durable_io import verify_failure_manifest
from src.qwen9b_ordinal_continuity_controller import partition_identity
from src.qwen9b_ordinal_continuity_training import (
    TrainingPlan, databricks_a10_plan)
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qualification_technical_successor import (
    SUCCESSOR_IDENTITY, authority, validate,
)


class ParentMlflowJournalTests(unittest.TestCase):
    @staticmethod
    def _parent_recovery_authority():
        commit = "a" * 40
        manifest = {"manifest_sha256": "b" * 64,
                    "source_commit": commit, "files": []}
        authority = {
            "version": PARENT_RECOVERY_AUTHORITY_VERSION,
            "execution_platform": "databricks", "cpu_only": True,
            "one_use": True, "automatic_retries": 0,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            "expires_at_utc": (
                datetime.now(timezone.utc) + timedelta(minutes=30)
            ).isoformat(),
            "identity": _V7_RECOVERY_ORIGINAL["identity"],
            "original": copy.deepcopy(_V7_RECOVERY_ORIGINAL),
            "recovery_source": {
                "source_commit": commit,
                "source_release_path": (
                    "/Workspace/Shared/research4/source_releases/" + commit),
                "source_sha256": manifest["manifest_sha256"],
                "source_manifest": manifest,
            },
            "files": {name: {"path": (
                                  "/Volumes/research4_simulated_continuity/"
                                  "bronze/model_data/recovery_inputs/"
                                  f"{name}.json"),
                              "sha256": "c" * 64}
                      for name in (
                          "intent", "plan", "base_qualification_receipt",
                          "failure_evidence", "task_output", "child_audit",
                          "mlflow_gold_audit")},
            "scope": {
                "allowed_task": "finalize_partition_mlflow",
                "max_cloud_runs": 1, "max_retries": 0,
                "timeout_seconds": 900, "gpu_execution": False,
                "model_loads": 0, "model_forwards": 0,
                "backward_passes": 0, "optimizer_updates": 0,
                "new_mlflow_run": False, "new_scientific_identity": False,
            },
            "human_authority": {
                "thread_id": "01a182ea-a06d-7223-9a17-90f33b1276f6",
                "message_id": "01a18df6-7e5a-7a02-8eb3-efab4055294e",
                "instruction": "try again just the failed step not all the steps",
            },
        }
        authority["authority_sha256"] = digest(authority)
        return authority

    def test_parent_recovery_authority_is_cpu_only_and_source_bound(self):
        authority = self._parent_recovery_authority()
        expected_root = Path(authority["recovery_source"][
            "source_release_path"])
        manifest = authority["recovery_source"]["source_manifest"]
        with patch(
                "src.databricks_continuity_partition.validate_source_manifest",
                return_value=manifest), patch(
                    "src.databricks_continuity_partition.build_source_manifest",
                    return_value=manifest), patch(
                        "src.databricks_continuity_partition.Path.resolve",
                        return_value=expected_root):
            self.assertEqual(
                validate_parent_recovery_authority(
                    authority, recovery_code_root=expected_root)[
                        "authority_sha256"],
                authority["authority_sha256"])
            changed = copy.deepcopy(authority)
            changed["original"]["owner_run_id"] = "1"
            changed["authority_sha256"] = digest({
                key: value for key, value in changed.items()
                if key != "authority_sha256"})
            with self.assertRaises(PermissionError):
                validate_parent_recovery_authority(
                    changed, recovery_code_root=expected_root)
            changed = copy.deepcopy(authority)
            changed["expires_at_utc"] = (
                datetime.fromisoformat(changed["issued_at_utc"])
                + timedelta(hours=2)).isoformat()
            changed["authority_sha256"] = digest({
                key: value for key, value in changed.items()
                if key != "authority_sha256"})
            with self.assertRaises(PermissionError):
                validate_parent_recovery_authority(
                    changed, recovery_code_root=expected_root)
            changed = copy.deepcopy(authority)
            changed["human_authority"]["message_id"] = "fabricated"
            changed["authority_sha256"] = digest({
                key: value for key, value in changed.items()
                if key != "authority_sha256"})
            with self.assertRaises(PermissionError):
                validate_parent_recovery_authority(
                    changed, recovery_code_root=expected_root)
            changed = copy.deepcopy(authority)
            changed["scope"]["model_forwards"] = 1
            changed["authority_sha256"] = digest({
                key: value for key, value in changed.items()
                if key != "authority_sha256"})
            with self.assertRaises(PermissionError):
                validate_parent_recovery_authority(
                    changed, recovery_code_root=expected_root)

    def test_parent_recovery_result_is_immutable_and_zero_execution(self):
        authority = self._parent_recovery_authority()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority_path = root / "AUTHORITY.json"
            authority_path.write_text(json.dumps(authority), encoding="utf-8")
            recovery = {
                "status": "parent_evidence_recovered",
                "authority_sha256": authority["authority_sha256"],
                "identity": authority["identity"],
                "original_owner_run_id": "303633587077489",
                "recovery_execution_run_id": "111",
                "mlflow_run_id": "d" * 32,
                "recovery_root": str(
                    root / "volume" / "parent_finalization_recovery" /
                    (PARENT_RECOVERY_AUTHORITY_VERSION + "-" +
                     authority["identity"])),
                "parent_finalization_sha256": "e" * 64,
                "parent_manifest_sha256": "f" * 64,
            }
            finalization = {
                "status": "complete", "identity": authority["identity"],
                "mlflow_run_id": "d" * 32,
                "receipt_sha256": "1" * 64,
                "journal_final_sha256": "2" * 64,
            }
            with patch(
                    "src.databricks_continuity_partition.VOLUME",
                    str(root / "volume")):
                first = publish_parent_recovery_result(
                    authority_path, recovery, finalization)
                second = publish_parent_recovery_result(
                    authority_path, recovery, finalization)
            self.assertEqual(first, second)
            self.assertEqual(
                {key: first[key] for key in (
                    "model_loads", "model_forwards", "backward_passes",
                    "optimizer_updates")},
                {"model_loads": 0, "model_forwards": 0,
                 "backward_passes": 0, "optimizer_updates": 0})
            changed = copy.deepcopy(recovery)
            changed["recovery_root"] = str(root / "attacker-controlled")
            with patch(
                    "src.databricks_continuity_partition.VOLUME",
                    str(root / "volume")), self.assertRaises(PermissionError):
                publish_parent_recovery_result(
                    authority_path, changed, finalization)

    def test_parent_recovery_route_cannot_execute_model_or_optimizer(self):
        source = inspect.getsource(recover_parent_evidence)
        for forbidden in (
                "_stage_local_model", "subprocess", "torch", ".backward(",
                ".step(", "snapshot_download", "run_partition("):
            self.assertNotIn(forbidden, source)
        self.assertIn("_replay_completed_child(", source)
        self.assertIn("_publish_parent_evidence(", source)
        self.assertIn('mlflow_run.info.status != "RUNNING"', source)
        self.assertIn("or gold_rows", source)

    def test_parent_recovery_replays_exact_prestate_before_publication(self):
        authority = self._parent_recovery_authority()
        original = authority["original"]
        files = authority["files"]
        files["intent"]["sha256"] = original["intent_file_sha256"]
        files["plan"]["sha256"] = original["plan_file_sha256"]
        files["failure_evidence"]["sha256"] = original[
            "failure_evidence_file_sha256"]
        files["task_output"]["sha256"] = original[
            "task_output_file_sha256"]
        files["child_audit"]["sha256"] = original[
            "child_audit_file_sha256"]
        files["mlflow_gold_audit"]["sha256"] = original[
            "mlflow_gold_audit_file_sha256"]
        authority["authority_sha256"] = digest({
            key: value for key, value in authority.items()
            if key != "authority_sha256"})
        plan = databricks_a10_plan(TrainingPlan())
        intent = {
            "identity": original["identity"],
            "source_commit": original["source_commit"],
            "source_sha256": original["source_manifest_sha256"],
            "intent_sha256": original["intent_sha256"],
            "plan_sha256": original["plan_sha256"],
            "global_journal_sha256": original["global_journal_sha256"],
            "model_tree_sha256": original["model_tree_sha256"],
            "model_receipt_sha256": original["model_receipt_sha256"],
            "base_qualification_receipt_sha256": original[
                "base_qualification_receipt_sha256"],
            "plan": asdict(plan),
        }
        base = {
            "identity": "base", "receipt_sha256": original[
                "base_qualification_receipt_sha256"],
            "plan_sha256": "1" * 64, "raw_results_sha256": "2" * 64,
            "base_qualification_intent_sha256": "3" * 64,
        }
        terminal = {"exit_code": 0, "operator_stop_sent": False,
                    "ceiling_kill_sent": False,
                    "child_exit_confirmed": True}
        failure = {
            "version": "research4-v7-parent-finalization-failure-evidence-v1",
            "source_commit": original["source_commit"],
            "owner_run_id": original["owner_run_id"],
            "gpu_task_run_id": original["gpu_task_run_id"],
            "error": "AttributeError: 'dict' object has no attribute 'optimizer_updates'",
            "source_file": "src/databricks_continuity_partition.py",
            "line_numbers": [427, 432, 433],
            "task_output_file_sha256": original["task_output_file_sha256"],
        }
        failure["failure_evidence_sha256"] = digest(failure)
        child = {
            "receipt": {"receipt_sha256": original[
                "training_receipt_sha256"]},
            "rank_receipt": {"receipt_sha256": original[
                "training_receipt_sha256"]},
            "evidence_sha256": original["evidence_sha256"],
            "journal": {"final": {"final_sha256": original[
                "journal_final_sha256"]}},
            "checkpoint_tree_sha256": original["checkpoint_tree_sha256"],
        }

        class Column:
            def __eq__(self, other):
                return self

            def __and__(self, other):
                return self

        functions = ModuleType("pyspark.sql.functions")
        functions.col = lambda name: Column()
        sql = ModuleType("pyspark.sql"); sql.functions = functions
        pyspark = ModuleType("pyspark"); pyspark.sql = sql

        class Table:
            def __init__(self, rows):
                self.rows = rows

            def where(self, expression):
                return self

            def collect(self):
                return self.rows

        class Spark:
            gold_calls = 0

            def table(self, name):
                if name == GOLD_ADMISSION_TABLE:
                    return Table([{"status": "released"}])
                if name == GOLD_RUN_TABLE:
                    self.gold_calls += 1
                    return Table([{"base": True}] if self.gold_calls == 1 else [])
                raise AssertionError(name)

        tags = {
            "mlflow.runName": original["identity"],
            "research4.parent_owned_run": "true",
            "research4.owner_run_id": original["owner_run_id"],
            "research4.intent_sha256": original["intent_sha256"],
            "research4.partition_identity": original["identity"],
            "research4.plan_sha256": original["plan_sha256"],
        }
        mlflow_run = SimpleNamespace(
            info=SimpleNamespace(
                experiment_id=original["mlflow_experiment_id"],
                status="RUNNING"),
            data=SimpleNamespace(tags=tags, params={}, metrics={}))
        client = SimpleNamespace(
            get_run=lambda run_id: mlflow_run,
            get_experiment=lambda experiment_id: SimpleNamespace(
                name=original["mlflow_experiment_path"]),
            list_artifacts=lambda run_id: [])
        mlflow = ModuleType("mlflow")
        mlflow.tracking = SimpleNamespace(MlflowClient=lambda: client)

        file_hashes = {row["path"]: row["sha256"]
                       for row in files.values()}
        def fake_file_sha(path):
            value = str(path).replace("\\", "/")
            if value in file_hashes:
                return file_hashes[value]
            suffixes = {
                "/records.jsonl": original["records_file_sha256"],
                "/TERMINAL.json": original["terminal_file_sha256"],
                "/output/CHILD_OUTPUT_MANIFEST.json": original[
                    "child_manifest_file_sha256"],
                "/output/TRAINING_RECEIPT.json": original[
                    "training_receipt_file_sha256"],
                "/output/RANK_0_RECEIPT.json": original[
                    "rank_receipt_file_sha256"],
                "/output/DATABRICKS_EVIDENCE.json": original[
                    "evidence_file_sha256"],
                "/output/checkpoint/delta1.pt": original[
                    "checkpoint_file_sha256"],
                "/output/optimizer_rank0.pt": original[
                    "optimizer_file_sha256"],
            }
            return next(expected for suffix, expected in suffixes.items()
                        if value.endswith(suffix))

        closeout_receipt = [original["closeout_receipt_sha256"]]

        def fake_read(path):
            value = str(path).replace("\\", "/")
            if value == "/Volumes/research4_simulated_continuity/bronze/model_data/recovery_inputs/authority.json":
                return authority
            if value == files["intent"]["path"]:
                return intent
            if value == files["plan"]["path"]:
                return asdict(plan)
            if value == files["base_qualification_receipt"]["path"]:
                return base
            if value.endswith("/TERMINAL.json"):
                return terminal
            if "/locks/consumed/" in value:
                return {"owner": original["owner_run_id"]}
            if "/locks/closeouts/" in value:
                return {"receipt_sha256": closeout_receipt[0]}
            raise AssertionError(value)

        def fake_self_hashed(path, **kwargs):
            return (failure if kwargs.get("field")
                    == "failure_evidence_sha256" else {})

        events = []
        published = {
            "parent_finalization": {"parent_finalization_sha256": "4" * 64},
            "parent_manifest": {"manifest_sha256": "5" * 64},
        }
        original_root = Path(
            "/Workspace/Shared/research4/source_releases/" +
            original["source_commit"])
        with patch.dict(sys.modules, {
                "pyspark": pyspark, "pyspark.sql": sql,
                "pyspark.sql.functions": functions, "mlflow": mlflow}), \
                patch(
                    "src.databricks_continuity_partition.validate_parent_recovery_authority",
                    return_value=authority), patch(
                    "src.databricks_continuity_partition.Path.resolve",
                    return_value=original_root), patch(
                    "src.databricks_continuity_partition._file_sha256",
                    side_effect=fake_file_sha), patch(
                    "src.databricks_continuity_partition._read",
                    side_effect=fake_read), patch(
                    "src.databricks_continuity_partition.validate_intent",
                    return_value=intent), patch(
                    "src.databricks_continuity_partition.verify_deployed_source",
                    return_value={
                        "manifest_sha256": original["source_manifest_sha256"],
                        "files": [{
                            "path": "src/databricks_continuity_partition.py",
                            "sha256": original[
                                "partition_source_file_sha256"]}]}), patch(
                    "src.databricks_continuity_partition.validate_base_qualification_receipt",
                    return_value=base), patch(
                    "src.databricks_continuity_partition.verify_intent_base_qualification_authority",
                    return_value={"source_commit": original["source_commit"]}), patch(
                    "src.databricks_continuity_partition.verify_model_receipt_authority",
                    return_value={
                        "tree_sha256": original["model_tree_sha256"],
                        "receipt_sha256": original["model_receipt_sha256"]}), patch(
                    "src.databricks_continuity_partition.validate_exact_gold_row"), patch(
                    "src.databricks_continuity_partition._verify_self_hashed_evidence",
                    side_effect=fake_self_hashed), patch(
                    "src.databricks_continuity_partition._read_exact_text",
                    return_value=failure["error"]), patch(
                    "src.databricks_continuity_partition._jsonl",
                    return_value=[{}] * original["records_count"]), patch(
                    "src.databricks_continuity_partition._replay_completed_child",
                    return_value=child), patch(
                    "src.databricks_continuity_partition._verify_finalizer_owner",
                    return_value="after_cleanup"), patch(
                    "src.databricks_continuity_partition._atomic_json",
                    side_effect=lambda *args, **kwargs: events.append("claim")), patch(
                    "src.databricks_continuity_partition._publish_parent_evidence",
                    side_effect=lambda **kwargs: (events.append("publish") or published)):
            value = recover_parent_evidence(
                spark=Spark(),
                authority_path=(
                    "/Volumes/research4_simulated_continuity/bronze/model_data/"
                    "recovery_inputs/authority.json"),
                recovery_code_root=authority["recovery_source"][
                    "source_release_path"],
                original_code_root=str(original_root),
                intent_path=files["intent"]["path"],
                plan_path=files["plan"]["path"],
                base_qualification_receipt_path=files[
                    "base_qualification_receipt"]["path"],
                original_owner_run_id=original["owner_run_id"],
                recovery_execution_run_id="111",
                recovery_job_id=original["job_id"],
                recovery_task_key="finalize_partition_mlflow")
            self.assertEqual(events, ["claim", "publish"])
            self.assertEqual(value["status"], "parent_evidence_recovered")
            with self.assertRaises(PermissionError):
                recover_parent_evidence(
                    spark=Spark(),
                    authority_path=(
                        "/Volumes/research4_simulated_continuity/bronze/model_data/"
                        "recovery_inputs/authority.json"),
                    recovery_code_root=authority["recovery_source"][
                        "source_release_path"],
                    original_code_root=str(original_root),
                    intent_path=files["intent"]["path"],
                    plan_path=files["plan"]["path"],
                    base_qualification_receipt_path=files[
                        "base_qualification_receipt"]["path"],
                    original_owner_run_id=original["owner_run_id"],
                    recovery_execution_run_id="112",
                    recovery_job_id="wrong",
                    recovery_task_key="finalize_partition_mlflow")
            closeout_receipt[0] = "0" * 64
            with self.assertRaisesRegex(PermissionError, "release state"):
                recover_parent_evidence(
                    spark=Spark(),
                    authority_path=(
                        "/Volumes/research4_simulated_continuity/bronze/model_data/"
                        "recovery_inputs/authority.json"),
                    recovery_code_root=authority["recovery_source"][
                        "source_release_path"],
                    original_code_root=str(original_root),
                    intent_path=files["intent"]["path"],
                    plan_path=files["plan"]["path"],
                    base_qualification_receipt_path=files[
                        "base_qualification_receipt"]["path"],
                    original_owner_run_id=original["owner_run_id"],
                    recovery_execution_run_id="113",
                    recovery_job_id=original["job_id"],
                    recovery_task_key="finalize_partition_mlflow")
            self.assertEqual(events, ["claim", "publish"])

    def test_completed_child_replay_accepts_dict_plan_and_binds_two_updates(self):
        plan = databricks_a10_plan(TrainingPlan())
        intent = {"identity": "v7", "intent_sha256": "1" * 64,
                  "plan_sha256": plan.full_plan_sha256,
                  "global_journal_sha256": "2" * 64,
                  "source_commit": "3" * 40,
                  "plan": asdict(plan),
                  "model_tree_sha256": "5" * 64,
                  "model_receipt_sha256": "8" * 64,
                  "base_qualification_receipt_sha256": "6" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "output"
            metadata_path = output / "checkpoint" / "adapter_config.json"
            metadata_path.parent.mkdir(parents=True)
            metadata = {"rank": plan.rank}
            metadata["receipt_sha256"] = digest(metadata)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            rank = {"status": "complete", "successful_optimizer_updates": 2,
                    "receipt_sha256": "7" * 64}
            receipt = {**rank, "checkpoint": metadata,
                       "checkpoint_metadata_sha256": hashlib.sha256(
                           metadata_path.read_bytes()).hexdigest()}
            evidence = {"value": 1}
            evidence["evidence_sha256"] = digest(evidence)
            for name, value in (("TRAINING_RECEIPT.json", receipt),
                                ("RANK_0_RECEIPT.json", rank),
                                ("DATABRICKS_EVIDENCE.json", evidence)):
                (output / name).write_text(json.dumps(value), encoding="utf-8")
            with patch(
                    "src.databricks_continuity_partition.verify_manifest") as manifest, \
                    patch(
                        "src.databricks_continuity_partition.validate_mlflow_journal",
                        return_value={"final": {"final_sha256": "3" * 64}}), \
                    patch(
                        "src.databricks_continuity_partition.verify_rank_receipt"), \
                    patch(
                        "src.databricks_continuity_partition.adjacent_pair_schedule",
                        return_value=[]), \
                    patch(
                        "src.databricks_continuity_partition.directory_tree_sha256",
                        return_value="4" * 64):
                value = _replay_completed_child(
                    root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[])
            self.assertEqual(value["receipt"]["successful_optimizer_updates"], 2)
            self.assertEqual(manifest.call_args.kwargs["required_update_count"], 2)
            self.assertEqual(
                manifest.call_args.kwargs["allowed_extra"],
                set())
            self.assertEqual(manifest.call_args.kwargs["expected_metadata"][
                "executed_optimizer_step_count"], 2)
            (root / "records.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "TERMINAL.json").write_text(
                json.dumps({"exit_code": 0, "operator_stop_sent": False,
                            "ceiling_kill_sent": False,
                            "child_exit_confirmed": True}),
                encoding="utf-8")
            (output / "CHILD_OUTPUT_MANIFEST.json").write_text(
                "{}\n", encoding="utf-8")
            with patch(
                    "src.databricks_continuity_partition._replay_completed_child",
                    return_value=value):
                published = _publish_parent_evidence(
                    run_root=root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[],
                    model_receipt={"tree_sha256": "5" * 64,
                                   "receipt_sha256": "8" * 64},
                    base_qualification={"receipt_sha256": "6" * 64},
                    primary_authority=None, parent_checkpoint=None,
                    owner_run_id="123", mlflow_run_id="9" * 32)
            self.assertEqual(published["status"], "pending_cpu_finalize")
            self.assertEqual(
                published["parent_finalization"]["pending_gold_row"][
                    "optimizer_updates"], 2)
            self.assertTrue((output / "PARENT_FINALIZATION.json").is_file())
            self.assertTrue((root / "PARENT_OUTPUT_MANIFEST.json").is_file())
            with patch(
                    "src.databricks_continuity_partition.verify_manifest") as replay_manifest, \
                    patch(
                        "src.databricks_continuity_partition.validate_mlflow_journal",
                        return_value={"final": {"final_sha256": "3" * 64}}), \
                    patch(
                        "src.databricks_continuity_partition.verify_rank_receipt"), \
                    patch(
                        "src.databricks_continuity_partition.adjacent_pair_schedule",
                        return_value=[]), \
                    patch(
                        "src.databricks_continuity_partition.directory_tree_sha256",
                        return_value="4" * 64):
                _replay_completed_child(
                    root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[])
            self.assertEqual(
                replay_manifest.call_args.kwargs["allowed_extra"],
                {"PARENT_FINALIZATION.json"})
            with patch(
                    "src.databricks_continuity_partition._replay_completed_child",
                    return_value=value):
                exact = _publish_parent_evidence(
                    run_root=root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[],
                    model_receipt={"tree_sha256": "5" * 64,
                                   "receipt_sha256": "8" * 64},
                    base_qualification={"receipt_sha256": "6" * 64},
                    primary_authority=None, parent_checkpoint=None,
                    owner_run_id="123", mlflow_run_id="9" * 32)
            self.assertEqual(
                exact["parent_manifest"]["manifest_sha256"],
                published["parent_manifest"]["manifest_sha256"])
            parent_bytes = (output / "PARENT_FINALIZATION.json").read_bytes()
            (output / "PARENT_FINALIZATION.json").unlink()
            with self.assertRaisesRegex(
                    PermissionError, "manifest exists without"):
                _publish_parent_evidence(
                    run_root=root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[],
                    model_receipt={"tree_sha256": "5" * 64,
                                   "receipt_sha256": "8" * 64},
                    base_qualification={"receipt_sha256": "6" * 64},
                    primary_authority=None, parent_checkpoint=None,
                    owner_run_id="123", mlflow_run_id="9" * 32)
            (output / "PARENT_FINALIZATION.json").write_bytes(parent_bytes)
            (root / "PARENT_OUTPUT_MANIFEST.json").unlink()
            with patch(
                    "src.databricks_continuity_partition._replay_completed_child",
                    return_value=value):
                replayed = _publish_parent_evidence(
                    run_root=root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[],
                    model_receipt={"tree_sha256": "5" * 64,
                                   "receipt_sha256": "8" * 64},
                    base_qualification={"receipt_sha256": "6" * 64},
                    primary_authority=None, parent_checkpoint=None,
                    owner_run_id="123", mlflow_run_id="9" * 32)
            self.assertEqual(
                replayed["parent_finalization"]["parent_finalization_sha256"],
                published["parent_finalization"]["parent_finalization_sha256"])
            (output / "PARENT_FINALIZATION.json").write_text(
                "{}\n", encoding="utf-8")
            with patch(
                    "src.databricks_continuity_partition._replay_completed_child",
                    return_value=value), self.assertRaises(FileExistsError):
                _publish_parent_evidence(
                    run_root=root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[],
                    model_receipt={"tree_sha256": "5" * 64,
                                   "receipt_sha256": "8" * 64},
                    base_qualification={"receipt_sha256": "6" * 64},
                    primary_authority=None, parent_checkpoint=None,
                    owner_run_id="123", mlflow_run_id="9" * 32)
            receipt["successful_optimizer_updates"] = 1
            (output / "TRAINING_RECEIPT.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            with patch("src.databricks_continuity_partition.verify_manifest"), \
                    self.assertRaises(PermissionError):
                _replay_completed_child(
                    root, intent=intent, plan_dict=asdict(plan),
                    exact_plan=plan, rows=[])
        source = inspect.getsource(run_partition)
        self.assertNotIn("plan.optimizer_updates", source)
        self.assertIn("_publish_parent_evidence", source)
        self.assertIn(
            "_replay_completed_child",
            inspect.getsource(_publish_parent_evidence))

    def test_early_model_preflight_failure_publishes_failure_manifest(self):
        plan = databricks_a10_plan(TrainingPlan())
        intent = {
            "plan": asdict(plan), "identity": SUCCESSOR_IDENTITY,
            "intent_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); intent_path = root / "intent.json"
            plan_path = root / "plan.json"
            intent_path.write_text(json.dumps(intent), encoding="utf-8")
            plan_path.write_text(json.dumps(asdict(plan)), encoding="utf-8")
            args = SimpleNamespace(
                intent=str(intent_path), plan=str(plan_path),
                output=str(root / "local"),
                durable_output=str(root / "volume"),
                model_receipt=str(root / "missing-model-receipt.json"),
                records=str(root / "records.jsonl"), admission=None,
                parent_checkpoint=None)
            with patch(
                    "src.databricks_continuity_partition_worker.validate_intent",
                    return_value=intent), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                        PermissionError,
                        "Per-run verified local model cache is required"):
                    run_partition_worker(args)
            failure = verify_failure_manifest(
                root / "volume", expected_metadata={
                    "identity": SUCCESSOR_IDENTITY,
                    "intent_sha256": "a" * 64,
                    "plan_sha256": plan.full_plan_sha256})
            self.assertEqual(failure["error_type"], "PermissionError")

    def fixture(self, root):
        plan = databricks_a10_plan(TrainingPlan())
        technical = authority()
        intent = {
            "intent_sha256": "1" * 64,
            "identity": partition_identity(plan),
            "source_manifest": {"manifest_sha256": "2" * 64,
                                "file_count": 10},
            "qualification_technical_successor_sha256": technical[
                "technical_successor_sha256"],
        }
        params = {
            "version": MLFLOW_JOURNAL_VERSION,
            "intent_sha256": intent["intent_sha256"],
            "partition_identity": intent["identity"],
            "plan_sha256": plan.full_plan_sha256,
            "model_revision": plan.model_revision,
            "optimizer_updates": plan.optimizer_updates,
            "rank": plan.rank, "alpha": plan.alpha,
            "source_manifest_sha256": "2" * 64,
            "source_manifest_file_count": 10,
            "technical_successor_sha256": technical[
                "technical_successor_sha256"],
            "numerical_policy": plan.numerical_policy,
            "grad_scaler_initial_scale": plan.grad_scaler_initial_scale,
            "grad_scaler_backoff_factor": plan.grad_scaler_backoff_factor,
            "max_overflow_adjustments": plan.max_overflow_adjustments,
            "minimum_grad_scaler_scale": plan.minimum_grad_scaler_scale,
        }
        params["params_sha256"] = _digest(params)
        (root / "MLFLOW_PARAMS.json").write_text(json.dumps(params))
        updates = []
        for step in range(plan.optimizer_updates):
            update = {"optimizer_update": step, "loss": 1.0 - step * .1,
                      "sft": .5, "preference": .2,
                      "cross_danger_margin": .1, "level0_kl_anchor": .05,
                      "gradient_norm_before_clip": .9,
                      "grad_scaler_before": 1.0,
                      "grad_scaler_after": 1.0,
                      "overflow_adjustments": 0,
                      "logical_tokens": 1024,
                      "replay_physical_tokens": 0,
                      "physical_tokens_processed": 1024,
                      "optimizer_step_pre_hook_calls": 1,
                      "minimum_grad_scaler_scale": 1.0,
                      "overflow_recovered": False,
                      "gradient_norm_positive": True,
                      "active_branch_finite_nonzero_gradient_element_count": 100,
                      "active_branch_finite_nonzero_gradient_density": 0.5,
                      "fixed_scale_no_growth": True,
                      "grad_scaler_attempts": [{
                          "attempt_index": 0,
                          "overflow": False,
                          "gradient_finite": True,
                          "scale_before": 1.0,
                          "scale_after": 1.0,
                          "optimizer_step_pre_hook_call_delta": 1,
                          "optimizer_step_executed": True,
                      }]}
            updates.append(update)
        rows = []
        for update in updates:
            row = {"version": MLFLOW_JOURNAL_VERSION,
                   "intent_sha256": intent["intent_sha256"],
                   "plan_sha256": plan.full_plan_sha256,
                   **{key: value for key, value in update.items()
                      if key != "grad_scaler_attempts"},
                   "grad_scaler_attempt_count": len(
                       update["grad_scaler_attempts"]),
                   "grad_scaler_attempts_sha256": _digest(
                       update["grad_scaler_attempts"])}
            row["progress_record_sha256"] = _digest(row); rows.append(row)
        progress = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                           for row in rows).encode()
        (root / "MLFLOW_PROGRESS.jsonl").write_bytes(progress)
        rank = {"updates": updates, "receipt_sha256": "3" * 64}
        receipt = {"receipt_sha256": "4" * 64, "resources": {
            "tokens_per_second": 12.5, "disk_free_bytes": 100,
            "gpu": {"max_allocated_bytes": 20, "max_reserved_bytes": 30,
                    "nvml": {"mean_utilization_percent": 80.0}}}}
        evidence = {"evidence_sha256": "5" * 64}
        artifacts = {
            "DATABRICKS_EVIDENCE.json": evidence,
            "RANK_0_RECEIPT.json": rank,
            "TRAINING_RECEIPT.json": receipt,
            "checkpoint/adapter_config.json": {"rank": plan.rank},
        }
        for relative, value in artifacts.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True))
        final = {"version": MLFLOW_JOURNAL_VERSION,
                 "params_sha256": params["params_sha256"],
                 "progress_sha256": hashlib.sha256(progress).hexdigest(),
                 "progress_count": len(rows),
                 "receipt_sha256": receipt["receipt_sha256"],
                 "rank_receipt_sha256": rank["receipt_sha256"],
                 "evidence_sha256": evidence["evidence_sha256"],
                 "artifact_paths": ["DATABRICKS_EVIDENCE.json",
                                    "RANK_0_RECEIPT.json",
                                    "TRAINING_RECEIPT.json",
                                    "checkpoint/adapter_config.json"],
                 "artifact_sha256": {
                     relative: hashlib.sha256(
                         (root / relative).read_bytes()).hexdigest()
                     for relative in artifacts
                 },
                 "mean_gpu_utilization_percent": 80.0,
                 "tokens_per_second": 12.5, "max_allocated_bytes": 20,
                 "max_reserved_bytes": 30, "disk_free_bytes": 100}
        final["final_sha256"] = _digest(final)
        (root / "MLFLOW_FINAL.json").write_text(json.dumps(final))
        return intent, plan, receipt, rank, evidence

    @staticmethod
    def terminal():
        return {"local_model_cache": {
            "copy_seconds": 2.0,
            "copy_bytes_per_second": 100.0,
            "local_disk_free_after": 500,
        }}

    def test_journal_replays_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = self.fixture(root)
            self.assertEqual(validate_mlflow_journal(
                root, intent=values[0], plan=values[1], receipt=values[2],
                rank_receipt=values[3], evidence=values[4])["final"][
                    "progress_count"], 2)
            rows = (root / "MLFLOW_PROGRESS.jsonl").read_text().splitlines()
            changed = json.loads(rows[0]); changed["loss"] = 99
            rows[0] = json.dumps(changed)
            (root / "MLFLOW_PROGRESS.jsonl").write_text("\n".join(rows) + "\n")
            with self.assertRaises(PermissionError):
                validate_mlflow_journal(
                    root, intent=values[0], plan=values[1], receipt=values[2],
                    rank_receipt=values[3], evidence=values[4])

    def test_progress_omission_extra_bool_and_nan_fail_even_if_rehashed(self):
        changes = {
            "omitted": lambda row: row.pop("loss"),
            "extra": lambda row: row.__setitem__("unreviewed", 1.0),
            "bool": lambda row: row.__setitem__("loss", True),
            "nan": lambda row: row.__setitem__("loss", float("nan")),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); values = self.fixture(root)
                rows = [json.loads(line) for line in (
                    root / "MLFLOW_PROGRESS.jsonl").read_text().splitlines()]
                row = rows[0]; row.pop("progress_record_sha256")
                change(row); row["progress_record_sha256"] = _digest(row)
                progress = "".join(
                    json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                    for item in rows).encode()
                (root / "MLFLOW_PROGRESS.jsonl").write_bytes(progress)
                final = json.loads((root / "MLFLOW_FINAL.json").read_text())
                final.pop("final_sha256")
                final["progress_sha256"] = hashlib.sha256(progress).hexdigest()
                final["final_sha256"] = _digest(final)
                (root / "MLFLOW_FINAL.json").write_text(json.dumps(final))
                with self.assertRaises(PermissionError):
                    validate_mlflow_journal(
                        root, intent=values[0], plan=values[1],
                        receipt=values[2], rank_receipt=values[3],
                        evidence=values[4])

    def test_artifact_byte_tamper_fails_even_with_unchanged_self_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); values = self.fixture(root)
            (root / "checkpoint" / "adapter_config.json").write_text(
                json.dumps({"rank": 99}))
            with self.assertRaises(PermissionError):
                validate_mlflow_journal(
                    root, intent=values[0], plan=values[1], receipt=values[2],
                    rank_receipt=values[3], evidence=values[4])

    def test_scalar_reconcile_repairs_missing_and_partial_failed_run(self):
        class Client:
            def __init__(self):
                self.params = {}; self.metrics = {}
                self.run = SimpleNamespace(
                    info=SimpleNamespace(status="FAILED"),
                    data=SimpleNamespace(params=self.params, tags={}))

            def get_run(self, _run_id):
                return self.run

            def log_param(self, _run_id, key, value):
                if key in self.params and self.params[key] != str(value):
                    raise PermissionError("conflicting parameter")
                self.params[key] = str(value)

            def log_metric(self, _run_id, key, value, step=0):
                self.metrics.setdefault(key, []).append(SimpleNamespace(
                    step=step, value=float(value)))

            def get_metric_history(self, _run_id, key):
                return list(self.metrics.get(key, ()))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); values = self.fixture(root)
            journal = validate_mlflow_journal(
                root, intent=values[0], plan=values[1], receipt=values[2],
                rank_receipt=values[3], evidence=values[4])
            client = Client()
            # Simulate a partial earlier publication followed by CPU repair.
            client.params["partition_identity"] = values[0]["identity"]
            client.metrics["loss"] = [SimpleNamespace(step=0, value=1.0)]
            result = _reconcile_mlflow_scalars(
                client, "run", journal, self.terminal(), write=True)
            self.assertEqual(client.params, result["params"])
            _reconcile_mlflow_scalars(
                client, "run", journal, self.terminal(), write=False)
            client.metrics["loss"].append(SimpleNamespace(step=0, value=9.0))
            with self.assertRaises(PermissionError):
                _reconcile_mlflow_scalars(
                    client, "run", journal, self.terminal(), write=False)

    def test_finished_scalar_verification_rejects_missing_state(self):
        class Client:
            def get_run(self, _run_id):
                return SimpleNamespace(data=SimpleNamespace(params={}, tags={}))

            def get_metric_history(self, _run_id, _key):
                return []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); values = self.fixture(root)
            journal = validate_mlflow_journal(
                root, intent=values[0], plan=values[1], receipt=values[2],
                rank_receipt=values[3], evidence=values[4])
            with self.assertRaises(PermissionError):
                _reconcile_mlflow_scalars(
                    Client(), "run", journal, self.terminal(), write=False)

    def test_completed_remote_state_rechecks_tags_scalars_and_artifacts(self):
        class Client:
            def __init__(self):
                self.params = {}; self.metrics = {}; self.artifacts = set(
                    _EXPECTED_MLFLOW_ARTIFACTS)
                self.tags = {}
                self.run = SimpleNamespace(
                    info=SimpleNamespace(status="FINISHED"),
                    data=SimpleNamespace(params=self.params, tags=self.tags))

            def get_run(self, _run_id):
                return self.run

            def log_param(self, _run_id, key, value):
                self.params[key] = str(value)

            def log_metric(self, _run_id, key, value, step=0):
                self.metrics.setdefault(key, []).append(SimpleNamespace(
                    step=step, value=float(value)))

            def get_metric_history(self, _run_id, key):
                return list(self.metrics.get(key, ()))

            def list_artifacts(self, _run_id, _path):
                return [SimpleNamespace(path=f"verified/{name}")
                        for name in self.artifacts]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); values = self.fixture(root)
            journal = validate_mlflow_journal(
                root, intent=values[0], plan=values[1], receipt=values[2],
                rank_receipt=values[3], evidence=values[4])
            client = Client()
            _reconcile_mlflow_scalars(
                client, "run", journal, self.terminal(), write=True)
            owner_tags = {"mlflow.runName": values[0]["identity"],
                          "research4.parent_owned_run": "true"}
            client.tags.update(owner_tags)
            client.tags.update({"terminal_status": "complete",
                                "cpu_artifact_finalize": "true"})
            _verify_completed_mlflow_state(
                client, "run", journal, self.terminal(), owner_tags)
            client.artifacts.remove("TERMINAL.json")
            with self.assertRaises(PermissionError):
                _verify_completed_mlflow_state(
                    client, "run", journal, self.terminal(), owner_tags)
            client.artifacts.add("TERMINAL.json")
            client.tags["cpu_artifact_finalize"] = "false"
            with self.assertRaises(PermissionError):
                _verify_completed_mlflow_state(
                    client, "run", journal, self.terminal(), owner_tags)

    def test_owner_binding_accepts_only_pre_or_post_cleanup_exact_states(self):
        owner = "123"; identity = SUCCESSOR_IDENTITY
        intent_sha = "a" * 64
        consumed = {"owner": owner, "identity": identity,
                    "intent_sha256": intent_sha,
                    "status": "reserved_after_gpu_ready_before_model"}
        consumed_row = SimpleNamespace(
            job_run_id=owner, status="consumed_after_gpu_ready_before_model")
        self.assertEqual(_verify_finalizer_owner(
            consumed, [consumed_row], owner_run_id=owner,
            identity=identity, intent_sha256=intent_sha,
            active=consumed, closeout=None), "before_gpu_closeout")
        closeout = {
            "version": "research4-databricks-active-reservation-closeout-v1",
            "owner": owner, "identity": identity,
            "intent_sha256": intent_sha,
            "terminal_status": "job_all_done_cleanup",
            "closed_at": "2026-10-29T00:00:00+00:00",
            "consumed_identity_retained": True,
            "reservation_sha256": digest(consumed),
        }
        closeout["receipt_sha256"] = digest(closeout)
        self.assertEqual(_verify_finalizer_owner(
            consumed, [consumed_row], owner_run_id=owner,
            identity=identity, intent_sha256=intent_sha,
            active=None, closeout=closeout),
            "after_gpu_closeout_before_cleanup")
        released_row = SimpleNamespace(
            job_run_id=owner,
            status="active_released_consumed_identity_retained")
        self.assertEqual(_verify_finalizer_owner(
            consumed, [released_row, consumed_row], owner_run_id=owner,
            identity=identity, intent_sha256=intent_sha,
            active=None, closeout=closeout), "after_cleanup")
        wrong = dict(consumed); wrong["owner"] = "wrong"
        with self.assertRaises(PermissionError):
            _verify_finalizer_owner(
                wrong, [consumed_row], owner_run_id=owner,
                identity=identity, intent_sha256=intent_sha,
                active=wrong, closeout=None)

    def test_parent_finalization_binds_terminal_receipts_and_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            run_root = Path(folder); output = run_root / "output"
            checkpoint = output / "checkpoint"; checkpoint.mkdir(parents=True)
            metadata_payload = {"version": "test", "rank": 4}
            metadata = {**metadata_payload,
                        "receipt_sha256": digest(metadata_payload)}
            metadata_path = checkpoint / "adapter_config.json"
            metadata_path.write_text(json.dumps(metadata, sort_keys=True))
            (checkpoint / "delta1.pt").write_bytes(b"checkpoint")
            rank = {"receipt_sha256": "3" * 64, "updates": []}
            receipt = {
                **rank, "checkpoint": metadata,
                "checkpoint_metadata_sha256": hashlib.sha256(
                    metadata_path.read_bytes()).hexdigest(),
            }
            evidence = {"evidence_sha256": "5" * 64}
            terminal = self.terminal()
            files = {
                "TRAINING_RECEIPT.json": receipt,
                "RANK_0_RECEIPT.json": rank,
                "DATABRICKS_EVIDENCE.json": evidence,
            }
            for name, value in files.items():
                (output / name).write_text(json.dumps(value, sort_keys=True))
            (run_root / "TERMINAL.json").write_text(
                json.dumps(terminal, sort_keys=True))
            identity = SUCCESSOR_IDENTITY
            plan = {"mode": "qualification", "condition": "self_continuity",
                    "stage": "delta1", "seed": 202610281,
                    "optimizer_updates": 2}
            intent = {"identity": identity, "intent_sha256": "1" * 64,
                      "plan_sha256": "2" * 64, "plan": plan,
                      "source_commit": "6" * 40,
                      "model_tree_sha256": "7" * 64,
                      "base_qualification_receipt_sha256": "8" * 64}
            journal = {"final": {"final_sha256": "4" * 64}}
            def sha(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            pending_gold = {
                "partition_identity": identity, "mode": plan["mode"],
                "condition": plan["condition"], "stage": plan["stage"],
                "seed": plan["seed"], "status": "pending_mlflow_finalize",
                "mlflow_run_id": "run-id", "plan_sha256": "2" * 64,
                "source_commit": "6" * 40,
                "model_tree_sha256": "7" * 64,
                "receipt_sha256": rank["receipt_sha256"],
                "intent_sha256": "1" * 64,
                "evidence_sha256": evidence["evidence_sha256"],
                "admission_sha256": None,
                "qualification_receipt_sha256": "8" * 64,
                "receipt_json": json.dumps(
                    receipt, sort_keys=True, separators=(",", ":")),
                "optimizer_updates": 2,
                "checkpoint_tree_sha256": directory_tree_sha256(checkpoint),
                "parent_checkpoint_tree_sha256": None,
                "receipt_replayed": True,
            }
            parent = {
                "version": "research4-parent-mlflow-finalization-v4",
                "identity": identity, "owner_run_id": "123",
                "intent_sha256": intent["intent_sha256"],
                "plan_sha256": intent["plan_sha256"],
                "mlflow_run_id": "run-id",
                "terminal_file_sha256": sha(run_root / "TERMINAL.json"),
                "journal_final_sha256": "4" * 64,
                "training_receipt_file_sha256": sha(
                    output / "TRAINING_RECEIPT.json"),
                "rank_receipt_file_sha256": sha(
                    output / "RANK_0_RECEIPT.json"),
                "evidence_file_sha256": sha(
                    output / "DATABRICKS_EVIDENCE.json"),
                "checkpoint_metadata_file_sha256": sha(metadata_path),
                "checkpoint_tree_sha256": directory_tree_sha256(checkpoint),
                "receipt_sha256": receipt["receipt_sha256"],
                "rank_receipt_sha256": rank["receipt_sha256"],
                "evidence_sha256": evidence["evidence_sha256"],
                "pending_gold_row": pending_gold,
            }
            parent["parent_finalization_sha256"] = digest(parent)
            (output / "PARENT_FINALIZATION.json").write_text(
                json.dumps(parent, sort_keys=True))
            self.assertEqual(_validate_parent_finalization(
                run_root, intent=intent, terminal=terminal, receipt=receipt,
                rank_receipt=rank, evidence=evidence, journal=journal,
                owner_run_id="123"), parent)
            (run_root / "TERMINAL.json").write_text("{}")
            with self.assertRaises(PermissionError):
                _validate_parent_finalization(
                    run_root, intent=intent, terminal={}, receipt=receipt,
                    rank_receipt=rank, evidence=evidence, journal=journal,
                    owner_run_id="123")

    def test_complete_parent_evidence_repairs_absent_gold_idempotently(self):
        expected = {
            "partition_identity": SUCCESSOR_IDENTITY,
            "intent_sha256": "1" * 64,
            "status": "pending_mlflow_finalize",
            "mlflow_run_id": "run-id",
            "receipt_sha256": "2" * 64,
        }
        parent = {"pending_gold_row": expected}
        stored = []
        read_rows = lambda: list(stored)
        append_row = lambda value: stored.append(dict(value))
        gold = _ensure_pending_gold_row(read_rows, append_row, parent)
        self.assertEqual(len(stored), 1)
        self.assertEqual(gold["receipt_sha256"], "2" * 64)
        again = _ensure_pending_gold_row(read_rows, append_row, parent)
        self.assertEqual(again, gold)
        self.assertEqual(len(stored), 1)
        stored[0]["receipt_sha256"] = "3" * 64
        with self.assertRaises(PermissionError):
            _ensure_pending_gold_row(read_rows, append_row, parent)

    def test_successor_authority_is_exact_and_tamper_fails(self):
        value = authority()
        self.assertEqual(validate(value), value)
        self.assertEqual(value["prior_identity"],
                         "qualification_rank4_a10_fp16_scale1_v6")
        self.assertEqual(value["successor_identity"],
                         "qualification_rank4_a10_volume_durable_v7")
        self.assertEqual(value["owner_run_id"], "598541262946494")
        self.assertEqual(
            value["prior_intent_sha256"],
            "96c19838047bf8e79ce1d48b572ee34ff6a7cbd5e8504830595ce665c1b717f3")
        self.assertEqual(
            value["closeout_receipt_sha256"],
            "63bc45c475030c5ffbae2f556a7c63dff2a2568c6d8fba13fbab857bc7d18cfa")
        self.assertEqual(
            value["predecessor_authority_sha256"],
            "16824b6af523b31979f88dd7aee5a48aa75f25e393299643f33fa33ad29e1edc")
        predecessor = value["predecessor_authority"]
        self.assertEqual(predecessor["owner_run_id"], "382017553734680")
        self.assertEqual(
            predecessor["predecessor_authority_sha256"],
            "e58a2d86b52bb6b06a8239847add8c8c84582503fbc4c8fd9ea32305467d6ea0")
        self.assertEqual(
            value["rank_partial_file_sha256"],
            "7ad986b86c74188088ada566295e1a6d072cff0a1bd8cd45f2cbc5dd3825f939")
        self.assertEqual(
            value["mlflow_progress_file_sha256"],
            "182c2c8302576ce3c41bc365823829b91e5eec5420f0c23cdb0431d9adb4a708")
        self.assertEqual(value["executed_optimizer_step_count"], 2)
        self.assertEqual(value["durable_published_update_count"], 1)
        self.assertFalse(value["checkpoint_written"])
        self.assertEqual(value["error"], "[Errno 29] Illegal seek")
        self.assertEqual(value["successor_execution_scope"],
                         "databricks-a10x1-only")
        self.assertEqual(value["successor_grad_scaler_scale"], 1.0)
        self.assertTrue(value["successor_storage_contract"][
            "per_update_immutable_receipt_before_advancement"])
        self.assertTrue(value["successor_storage_contract"][
            "checkpoint_optimizer_local_then_publish"])
        successor_plan = databricks_a10_plan(TrainingPlan())
        self.assertEqual(value["successor_plan_sha256"],
                         successor_plan.full_plan_sha256)
        self.assertRegex(value["technical_successor_sha256"], r"^[0-9a-f]{64}$")
        changed = copy.deepcopy(value); changed["executed_optimizer_step_count"] = 1
        with self.assertRaises(PermissionError):
            validate(changed)
        changed = copy.deepcopy(value)
        changed["predecessor_authority"]["optimizer_updates"] = 1
        with self.assertRaises(PermissionError):
            validate(changed)

    def test_child_has_no_mlflow_and_finalizer_orders_gold_first(self):
        root = Path(__file__).parents[1]
        child = (root / "src" / "databricks_continuity_partition_worker.py").read_text()
        parent = (root / "src" / "databricks_continuity_partition.py").read_text()
        self.assertNotIn("import mlflow", child)
        self.assertNotIn("MlflowClient", child)
        self.assertNotIn("mlflow-run-id", child)
        failure_index = child.index("def record_failure(exc):")
        self.assertLess(failure_index,
                        child.index("params = {", failure_index))
        self.assertLess(child.index("try:\n        _atomic_json"),
                        child.index("dependency = assert_fast_deltanet_dependencies"))
        self.assertGreaterEqual(child.count("record_failure(exc)"), 2)
        self.assertIn('"PYTORCH_CUDA_ALLOC_CONF": DATABRICKS_A10_ALLOCATOR_CONF',
                      parent)
        self.assertIn('os.environ.get("PYTORCH_CUDA_ALLOC_CONF")', child)
        self.assertIn('!= DATABRICKS_A10_ALLOCATOR_CONF', child)
        finalizer = parent[parent.index("def finalize_partition_mlflow"):]
        self.assertIn("consumed_path", finalizer)
        self.assertLess(finalizer.index("SET status = 'complete'"),
                        finalizer.index("set_terminated(mlflow_run_id"))


if __name__ == "__main__":
    unittest.main()
