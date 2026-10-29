"""CPU-only tests for immutable packaging and matrix orchestration."""
from pathlib import Path
from dataclasses import asdict
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.action_serialization import SERIALIZATION_CONTRACT_VERSION
from src.qwen9b_ordinal_continuity_controller import (
    AGGREGATE_VERSION,
    EXPERIMENT_VERSION,
    PARTITION_INTENT_VERSION,
    TERMINAL_VERSION,
    PartitionController,
    aggregate_terminal_receipts,
    exact_worker_argv,
    matrix_plans,
    partition_identity,
    qualification_plan,
    validate_experiment_spec,
    validate_partition_intent,
)
from src.qwen9b_ordinal_continuity_lora import MODEL_ID, MODEL_REVISION
from src.qwen9b_ordinal_continuity_training import TrainingPlan
from src.qwen9b_ordinal_continuity_runtime import (
    canonical_sha256,
    file_sha256,
    finalize_receipt,
    verify_package_manifest,
    verify_tokenizer_receipt,
)
from src.qwen9b_ordinal_continuity_worker import _prepare_pre_model_runtime


def spec():
    return {
        "version": EXPERIMENT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": ["self_continuity", "artifact_continuity", "label_only"],
        "seeds": [202610281, 202610282, 202610283],
        "stages": ["delta1", "delta2", "delta3"],
        "qualification_updates": 2,
        "primary_optimizer_updates": 144,
        "baseline": "frozen_base",
        "automatic_retries": 0,
        "one_use": True,
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
    }


class RuntimeBindingTests(unittest.TestCase):
    def test_exact_matrix_has_twenty_seven_residual_stages(self):
        plans = matrix_plans(spec(), 8)
        self.assertEqual(len(plans), 27)
        self.assertEqual({plan.rank for plan in plans}, {8})
        self.assertEqual({plan.optimizer_updates for plan in plans}, {144})
        self.assertEqual({plan.global_batch_size for plan in plans}, {4})
        self.assertEqual({plan.occurrence_count for plan in plans}, {576})
        self.assertEqual(qualification_plan(spec(), 4).optimizer_updates, 2)
        with self.assertRaises(ValueError):
            qualification_plan(spec(), 16)

    def test_experiment_spec_rejects_all_update_count_drift(self):
        accepted = validate_experiment_spec(spec())
        self.assertEqual(
            (accepted["qualification_updates"],
             accepted["primary_optimizer_updates"]),
            (2, 144))

        for value in (1, 3):
            changed = spec()
            changed["qualification_updates"] = value
            with self.subTest(field="qualification_updates", value=value):
                with self.assertRaisesRegex(PermissionError, "preregistered matrix"):
                    validate_experiment_spec(changed)

    def test_partition_intent_rejects_plan_level_update_drift(self):
        def intent_for(plan, parent=None):
            value = {
                "version": PARTITION_INTENT_VERSION, "one_use": True,
                "attempt_number": 1, "automatic_retries": 0,
                "partition_identity": partition_identity(plan),
                "plan_sha256": plan.full_plan_sha256,
                "wall_clock_stop_seconds": 3600,
                "wall_clock_ceiling_seconds": 4200,
                "expected_parent_checkpoint_tree_sha256": parent,
                "frozen_base_model_sha256": "b" * 64,
            }
            return value

        bad_plans = (
            TrainingPlan(mode="qualification", condition="self_continuity",
                         stage="delta1", rank=4, alpha=8.0,
                         seed=202610281, optimizer_updates=1).validate(),
            TrainingPlan(mode="primary", condition="self_continuity",
                         stage="delta1", rank=4, alpha=8.0,
                         seed=202610281, optimizer_updates=1).validate(),
        )
        for plan in bad_plans:
            parent = None
            with self.subTest(mode=plan.mode, stage=plan.stage):
                with self.assertRaisesRegex(PermissionError, "Executable partition exposure"):
                    validate_partition_intent(intent_for(plan, parent), plan, parent)

        for value in (1, 143, 145):
            changed = spec()
            changed["primary_optimizer_updates"] = value
            with self.subTest(field="primary_optimizer_updates", value=value):
                with self.assertRaisesRegex(PermissionError, "preregistered matrix"):
                    validate_experiment_spec(changed)

    def test_package_manifest_rejects_member_and_tree_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "research4" / "src" / "worker.py"
            source.parent.mkdir(parents=True)
            source.write_text("original\n", encoding="utf-8")
            files = [{"path": "research4/src/worker.py",
                      "sha256": file_sha256(source), "bytes": source.stat().st_size}]
            manifest = {
                "files": files,
                "source_paths": ["research4/src/worker.py"],
                "tree_sha256": canonical_sha256(files),
                "source_tree_sha256": canonical_sha256(files),
            }
            path = root / "PACKAGE_MANIFEST.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verified = verify_package_manifest(root, path)
            self.assertEqual(verified["verified_tree_sha256"], canonical_sha256(files))
            source.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "member differs"):
                verify_package_manifest(root, path)

    def test_durable_tokenizer_receipt_rejects_undeclared_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer_file = root / "tokenizer.json"
            tokenizer_file.write_text('{"model":"pinned"}\n', encoding="utf-8")
            files = [{"path": "tokenizer.json", "sha256": file_sha256(tokenizer_file),
                      "bytes": tokenizer_file.stat().st_size}]
            receipt = finalize_receipt({
                "version": "qwen35-9b-tokenizer-files-v1",
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "resolved_commit": MODEL_REVISION,
                "tokenizer_class": "PinnedTokenizer", "eos_token_id": 1,
                "pad_token_id": 1, "files": files,
                "tree_sha256": canonical_sha256(files),
            })
            receipt_path = root / "TOKENIZER_RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(
                verify_tokenizer_receipt(
                    root, receipt_path, model_id=MODEL_ID, revision=MODEL_REVISION
                )["receipt_sha256"], receipt["receipt_sha256"])
            (root / "undeclared.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "undeclared"):
                verify_tokenizer_receipt(
                    root, receipt_path, model_id=MODEL_ID, revision=MODEL_REVISION)

    def test_worker_pre_model_binding_rejects_actual_cli_tamper(self):
        plan = qualification_plan(spec(), 4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "research4" / "src" / "worker.py"
            source.parent.mkdir(parents=True)
            source.write_text("immutable\n", encoding="utf-8")
            files = [{"path": "research4/src/worker.py",
                      "sha256": file_sha256(source), "bytes": source.stat().st_size}]
            tree = canonical_sha256(files)
            manifest = {"files": files, "source_paths": [files[0]["path"]],
                        "tree_sha256": tree, "source_tree_sha256": tree}
            manifest_path = root / "PACKAGE_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            intent_path = root / "intent.json"
            output = root / "output"
            tokenizer_receipt = {"receipt_sha256": "t" * 64}
            dataset_receipt = {"receipt_sha256": "d" * 64}
            data = {"data": "bound"}
            exposure = {"exposure": "bound"}
            argv = ["torchrun", "--standalone", "--nproc_per_node=2", "-m",
                    "research4.src.qwen9b_ordinal_continuity_worker", "--exact"]
            intent = {
                "version": PARTITION_INTENT_VERSION,
                "one_use": True, "automatic_retries": 0, "attempt_number": 1,
                "plan_sha256": plan.full_plan_sha256,
                "output_root": str(output.resolve()),
                "worker_cli_sha256": canonical_sha256(argv),
                "data_binding_sha256": canonical_sha256(data),
                "exposure_binding_sha256": canonical_sha256(exposure),
                "dataset_receipt_sha256": dataset_receipt["receipt_sha256"],
                "source_sha256": tree, "package_tree_sha256": tree,
                "package_manifest_sha256": file_sha256(manifest_path),
                "tokenizer_receipt_sha256": tokenizer_receipt["receipt_sha256"],
                "parent_checkpoint_tree_sha256": None,
            }
            intent_path.write_text(json.dumps(intent), encoding="utf-8")
            args = SimpleNamespace(
                package_root=str(root), package_manifest=str(manifest_path),
                package_tree_sha256=tree, source_sha256=tree,
                generated_root=str(root / "generated"), tokenizer_root=str(root / "tok"),
                tokenizer_receipt=str(root / "tok" / "TOKENIZER_RECEIPT.json"),
                actual_worker_argv=argv, intent=str(intent_path),
                parent_checkpoint=None, admission=None)
            prepared = {"tokenizer_receipt": tokenizer_receipt,
                        "dataset_receipt": dataset_receipt,
                        "data_binding": data, "exposure_binding": exposure}
            with patch(
                    "src.qwen9b_ordinal_continuity_worker.prepare_runtime_bindings",
                    return_value=prepared):
                result = _prepare_pre_model_runtime(args, plan)
                self.assertEqual(result["summary"]["worker_cli_sha256"],
                                 canonical_sha256(argv))
                args.actual_worker_argv = argv + ["--tampered"]
                with self.assertRaisesRegex(PermissionError, "launch/input"):
                    _prepare_pre_model_runtime(args, plan)

    def test_cleanup_gate_requires_each_rank_release_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rank in (0, 1):
                payload = finalize_receipt({
                    "status": "released", "rank": rank,
                    "memory_allocated_bytes": 0, "ddp_destroyed": True,
                    "cuda_cache_cleared": True, "model_reference_deleted": True,
                    "scratch_absent_after_cleanup": True if rank == 0 else None,
                })
                (root / f"RANK_{rank}_POST_CLEANUP.json").write_text(
                    json.dumps(payload), encoding="utf-8")
            # The release gate includes CPU tokenizer/token payload release.
            for rank in (0, 1):
                value = json.loads((root / f"RANK_{rank}_POST_CLEANUP.json").read_text())
                value["tokenizer_and_tokenized_payloads_released"] = True
                value = finalize_receipt(value)
                (root / f"RANK_{rank}_POST_CLEANUP.json").write_text(json.dumps(value))
            self.assertTrue(PartitionController._cleanup_pass(root))
            changed = json.loads((root / "RANK_1_POST_CLEANUP.json").read_text())
            changed["memory_allocated_bytes"] = 1
            changed = finalize_receipt(changed)
            (root / "RANK_1_POST_CLEANUP.json").write_text(
                json.dumps(changed), encoding="utf-8")
            self.assertFalse(PartitionController._cleanup_pass(root))

    def test_partition_intent_and_offline_aggregate_bind_all_stage_chains(self):
        plans = matrix_plans(spec(), 4)
        base_sha = "b" * 64
        rows, prior = [], {}
        for plan in plans:
            key = (plan.condition, plan.seed)
            parent = prior.get(key)
            checkpoint = canonical_sha256([plan.full_plan_sha256])
            row = finalize_receipt({
                "version": TERMINAL_VERSION, "status": "complete",
                "plan": asdict(plan), "parent_checkpoint_tree_sha256": parent,
                "checkpoint_tree_sha256": checkpoint,
                "frozen_base_model_sha256": base_sha,
                "residual_lineage_pass": True,
                "post_cleanup_pass": True, "primary_save_reload_pass": True})
            rows.append(row)
            prior[key] = checkpoint
        receipt = aggregate_terminal_receipts(
            rows, frozen_base_model_sha256=base_sha)
        self.assertEqual(receipt["version"], AGGREGATE_VERSION)
        self.assertEqual(receipt["policy_terminal_count"], 27)
        with self.assertRaisesRegex(PermissionError, "incomplete"):
            aggregate_terminal_receipts(
                rows[:-1], frozen_base_model_sha256=base_sha)
        broken = [dict(row) for row in rows]
        broken[1]["parent_checkpoint_tree_sha256"] = "f" * 64
        broken[1].pop("receipt_sha256")
        broken[1] = finalize_receipt(broken[1])
        with self.assertRaisesRegex(PermissionError, "predecessor"):
            aggregate_terminal_receipts(
                broken, frozen_base_model_sha256=base_sha)

        policy = plans[0]
        intent = {
            "version": PARTITION_INTENT_VERSION, "one_use": True,
            "attempt_number": 1, "automatic_retries": 0,
            "partition_identity": partition_identity(policy),
            "plan_sha256": policy.full_plan_sha256,
            "wall_clock_stop_seconds": 3600,
            "wall_clock_ceiling_seconds": 4200,
            "expected_parent_checkpoint_tree_sha256": None,
            "frozen_base_model_sha256": base_sha}
        self.assertEqual(validate_partition_intent(
            intent, policy, None)["wall_clock_stop_seconds"], 3600)


if __name__ == "__main__":
    unittest.main()
