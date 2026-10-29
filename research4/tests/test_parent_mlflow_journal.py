import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.databricks_continuity_partition import (
    _EXPECTED_MLFLOW_ARTIFACTS,
    _ensure_pending_gold_row, _reconcile_mlflow_scalars,
    _validate_parent_finalization,
    _verify_completed_mlflow_state, _verify_finalizer_owner,
)
from src.databricks_simulated_continuity import digest
from src.databricks_continuity_partition_worker import (
    MLFLOW_JOURNAL_VERSION, _digest, validate_mlflow_journal,
)
from src.qwen9b_ordinal_continuity_controller import partition_identity
from src.qwen9b_ordinal_continuity_training import TrainingPlan
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qualification_technical_successor import (
    SUCCESSOR_IDENTITY, authority, validate,
)


class ParentMlflowJournalTests(unittest.TestCase):
    def fixture(self, root):
        plan = replace(
            TrainingPlan(), device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=1, accumulation_steps=4).validate()
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
        }
        params["params_sha256"] = _digest(params)
        (root / "MLFLOW_PARAMS.json").write_text(json.dumps(params))
        updates = []
        for step in range(plan.optimizer_updates):
            update = {"optimizer_update": step, "loss": 1.0 - step * .1,
                      "sft": .5, "preference": .2,
                      "cross_danger_margin": .1, "level0_kl_anchor": .05,
                      "gradient_norm_before_clip": .9,
                      "grad_scaler_before": 1.0, "grad_scaler_after": 1.0}
            updates.append(update)
        rows = []
        for update in updates:
            row = {"version": MLFLOW_JOURNAL_VERSION,
                   "intent_sha256": intent["intent_sha256"],
                   "plan_sha256": plan.full_plan_sha256, **update}
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
                "version": "research4-parent-mlflow-finalization-v1",
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
                         "qualification_rank4_gated_q_geometry_v3")
        self.assertEqual(value["successor_identity"],
                         "qualification_rank4_a10_microbatch1_v4")
        self.assertEqual(value["owner_run_id"], "739463343455961")
        self.assertEqual(
            value["prior_intent_sha256"],
            "3aced02640a00040a093fa78921953054ff4554b99b22c5e2e5a3233f7d4635b")
        self.assertEqual(
            value["closeout_receipt_sha256"],
            "b29476067c04b3d145729412f2a68e1b56b839f3e2ed22dd1d2f6347e3e168b6")
        self.assertEqual(
            value["predecessor_authority_sha256"],
            "f1c5d1ce2eeff6a9089a346db2a72ca11392805b0752539dc862bcb39c42a3bb")
        self.assertEqual(
            value["predecessor_authority"]["predecessor_authority_sha256"],
            "e029c82e61172517f3b94cd6d55897db14516bf2193c98381531aa3b50fcd335")
        self.assertEqual(
            value["rank_partial_file_sha256"],
            "6f66379b23aaf2a242537bcd3b7fb2c44c4944a8d69dc5a50afd6654546af843")
        self.assertEqual(
            value["mlflow_progress_file_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(value["progress_journal_entries"], 0)
        self.assertEqual(value["optimizer_updates"], 0)
        self.assertFalse(value["checkpoint_written"])
        self.assertEqual((value["prior_micro_batch_size"],
                          value["prior_accumulation_steps"]), (2, 2))
        self.assertEqual((value["successor_micro_batch_size"],
                          value["successor_accumulation_steps"]), (1, 4))
        self.assertEqual(value["qualification_occurrence_count"], 8)
        self.assertEqual(value["primary_occurrence_count"], 576)
        successor_plan = replace(
            TrainingPlan(), device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=1, accumulation_steps=4).validate()
        self.assertEqual(value["successor_plan_sha256"],
                         successor_plan.full_plan_sha256)
        self.assertEqual(value["successor_global_journal_sha256"],
                         value["prior_global_journal_sha256"])
        changed = copy.deepcopy(value); changed["optimizer_updates"] = 1
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
        self.assertIn('"PYTORCH_CUDA_ALLOC_CONF": DATABRICKS_A10_ALLOCATOR_CONF',
                      parent)
        self.assertIn(
            'os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != '
            'DATABRICKS_A10_ALLOCATOR_CONF', child)
        finalizer = parent[parent.index("def finalize_partition_mlflow"):]
        self.assertIn("consumed_path", finalizer)
        self.assertLess(finalizer.index("SET status = 'complete'"),
                        finalizer.index("set_terminated(mlflow_run_id"))


if __name__ == "__main__":
    unittest.main()
