import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continuity_transfer_evaluation import (
    BASELINE_CONDITION, CONDITIONS, TRAINING_SEEDS, digest,
    validate_unblinding_key,
)
from src.continuity_transfer_packaging import (
    PACKAGING_PLAN_VERSION, TRANSFER_PACKAGE_IDENTITY,
    TRANSFER_PACKAGE_PARENT, _build_training_audit,
    _publish_private_key, _publish_public_package,
    _publish_public_with_key_rollback,
    _claim_package_identity, _complete_package_identity,
    _sanitized_evaluation_config, _training_separation_receipt,
    assert_public_package_blinded, build_blinding_map,
    validate_gold_matrix, validate_packaging_plan,
)
from src.continuity_transfer_inference_worker import verify_checkpoint_manifest
from src.qwen9b_ordinal_continuity_lora import (
    MODEL_ID, MODEL_REVISION, OrdinalContinuityLoRAConfig,
)
from dataclasses import asdict
from src.databricks_simulated_continuity import (
    GOLD_SUMMARY_TABLE, SOURCE_MANIFEST_PATHS, SOURCE_MANIFEST_VERSION,
)


class TransferPackagingTests(unittest.TestCase):
    def source_manifest(self):
        files = [{"path": path, "sha256": "a" * 64, "canonical_bytes": 1}
                 for path in SOURCE_MANIFEST_PATHS]
        value = {
            "version": SOURCE_MANIFEST_VERSION, "source_commit": "b" * 40,
            "file_count": len(files), "total_canonical_bytes": len(files),
            "files": files,
        }
        value["manifest_sha256"] = digest(value)
        return value

    def plan(self):
        source = self.source_manifest()
        value = {
            "version": PACKAGING_PLAN_VERSION,
            "identity": TRANSFER_PACKAGE_IDENTITY,
            "execution_platform": "databricks", "cpu_only": True,
            "gpu_runs_started": 0, "model_forwards": 0,
            "optimizer_updates": 0, "automatic_retries": 0,
            "source_commit": "b" * 40, "source_manifest": source,
            "source_sha256": source["manifest_sha256"],
            "model_tree_sha256": "c" * 64,
            "base_qualification_receipt_path": "/Volumes/a/b/c/base.json",
            "base_qualification_receipt_sha256": "d" * 64,
            "base_qualification_receipt_file_sha256": "e" * 64,
            "capacity_qualification_receipt_path": "/Volumes/a/b/c/capacity.json",
            "capacity_qualification_receipt_sha256": "f" * 64,
            "capacity_qualification_receipt_file_sha256": "1" * 64,
            "gold_summary_table": GOLD_SUMMARY_TABLE,
            "expected_policy_partition_count": 27,
            "expected_adapted_endpoint_count": 9,
            "expected_frozen_base_endpoint_count": 1,
            "registry_seed": 2026102804,
            "package_parent": TRANSFER_PACKAGE_PARENT,
        }
        value["plan_sha256"] = digest(value)
        return value

    def test_packaging_plan_is_cpu_only_and_exact(self):
        plan = self.plan()
        self.assertEqual(validate_packaging_plan(plan), plan)
        for field, changed in (
            ("cpu_only", False), ("gpu_runs_started", 1),
            ("expected_policy_partition_count", 26),
            ("package_parent", "/tmp/packages"),
        ):
            bad = copy.deepcopy(plan); bad[field] = changed
            payload = dict(bad); payload.pop("plan_sha256")
            bad["plan_sha256"] = digest(payload)
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_packaging_plan(bad)

    def test_gold_matrix_fails_closed_without_exact_27_rows(self):
        base = {
            "source_commit": "b" * 40, "source_sha256": self.plan()["source_sha256"],
            "model_tree_sha256": "c" * 64, "receipt_sha256": "d" * 64,
        }
        with patch(
                "src.continuity_transfer_packaging.validate_base_qualification_receipt",
                return_value=base):
            with self.assertRaises(PermissionError):
                validate_gold_matrix(
                    plan=self.plan(), base_receipt=base,
                    capacity_receipt={}, gold_rows=[], summary_rows=[])

    def test_blinding_is_deterministic_complete_and_separate(self):
        mapping, key = build_blinding_map()
        mapping2, key2 = build_blinding_map()
        expected_cells = {(BASELINE_CONDITION, None)} | {
            (condition, seed) for condition in CONDITIONS for seed in TRAINING_SEEDS}
        self.assertEqual(set(mapping), expected_cells)
        self.assertEqual(len(set(mapping.values())), 10)
        self.assertEqual(validate_unblinding_key(key, set(mapping.values())), {
            blind: (row["condition"], row["training_seed"])
            for blind, row in key.items()
        })
        for blind in mapping.values():
            self.assertTrue(blind.startswith("blind-"))
            self.assertFalse(any(condition in blind for condition in CONDITIONS))
        self.assertTrue(set(mapping.values()).isdisjoint(mapping2.values()))
        self.assertNotEqual(key, key2)

    def test_training_separation_replays_all_exact_silver_records(self):
        rows = []
        for condition in CONDITIONS:
            for index in range(2304):
                rows.append({
                    "record_id": f"{condition}-{index:04d}",
                    "record_kind": condition, "condition": condition,
                    "split": "train", "family_id": f"family-{index % 8}",
                    "danger_level": index % 4,
                    "instruction": "Follow the simulator rules.",
                })
        receipt = _training_separation_receipt(rows, source_commit="b" * 40)
        self.assertEqual(receipt["training_record_count"], 6912)
        self.assertTrue(receipt["training_transfer_separation_passed"])
        self.assertEqual(receipt["transfer_action_labels_found"], 0)
        bad = copy.deepcopy(rows); bad[0]["instruction"] = "exploit the vulnerable fixture"
        with self.assertRaises(PermissionError):
            _training_separation_receipt(bad, source_commit="b" * 40)

    def test_training_audit_covers_the_three_stage_chain(self):
        chain = []
        parent = None
        for index, stage in enumerate(("delta1", "delta2", "delta3"), 1):
            tree = f"{index:064x}"
            chain.append({
                "identity": f"cell_{stage}", "stage": stage,
                "plan_sha256": f"{index + 10:064x}",
                "checkpoint_tree_sha256": tree,
                "parent_checkpoint_tree_sha256": parent,
                "training_receipt_sha256": f"{index + 20:064x}",
                "rank_receipt_sha256": f"{index + 30:064x}",
                "evidence_sha256": f"{index + 40:064x}",
                "optimizer_updates": 144,
            })
            parent = tree
        endpoint = {
            **chain[-1], "condition": "self_continuity", "seed": 202610281,
            "source_sha256": "8" * 64, "config_sha256": "9" * 64,
            "checkpoint_payload_members_sha256": "a" * 64,
        }
        audit = _build_training_audit(
            {"endpoint": endpoint, "chain": chain},
            matrix_sha256="b" * 64, source_commit="c" * 40)
        self.assertEqual(audit["cumulative_stage_optimizer_updates"], 432)
        self.assertEqual([row["stage"] for row in audit[
            "three_residual_stage_chain"]], ["delta1", "delta2", "delta3"])
        payload = dict(audit); embedded = payload.pop("audit_payload_sha256")
        self.assertEqual(embedded, digest(payload))

    def test_public_config_and_recursive_text_members_reveal_no_identity(self):
        original = {
            "config": {"model_id": "Qwen/Qwen3.5-9B", "model_revision": "r",
                       "rank": 4, "alpha": 8.0},
            "target_names": ["model.layers.0.q_proj"],
            "base_sha256": "1" * 64, "source_sha256": "2" * 64,
            "active_branches": ["delta1", "delta2", "delta3"],
            "branches": {name: {"file_sha256": str(index) * 64,
                                 "state_sha256": str(index + 3) * 64,
                                 "tensor_count": 2}
                         for index, name in enumerate(
                             ("delta1", "delta2", "delta3"), 1)},
            "adapter_sha256": "9" * 64,
            "condition": "self_continuity", "seed": 202610281,
            "plan_sha256": "a" * 64,
        }
        endpoint = {
            "config": original, "checkpoint_tree_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "checkpoint_payload_members_sha256": "d" * 64,
            "training_receipt_sha256": "e" * 64,
        }
        public = _sanitized_evaluation_config({"endpoint": endpoint})
        encoded = json.dumps(public)
        for field in (
            "condition", "seed", "plan_sha256", "original_checkpoint_tree_sha256",
            "original_checkpoint_config_sha256",
            "original_checkpoint_payload_members_sha256",
            "original_training_receipt_sha256", "original_training_audit_sha256",
        ):
            self.assertNotIn(field, public)
        for token in (*CONDITIONS, *(str(seed) for seed in TRAINING_SEEDS)):
            self.assertNotIn(token, encoded)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); nested = root / "nested"; nested.mkdir()
            (nested / "adapter_config.json").write_text(encoded)
            (root / "MANIFEST.json").write_text('{"blind":true}\n')
            self.assertTrue(assert_public_package_blinded(root)["blinded"])
            for token in (*CONDITIONS, *(str(seed) for seed in TRAINING_SEEDS)):
                (nested / "leak.txt").write_text(token)
                with self.subTest(token=token), self.assertRaises(PermissionError):
                    assert_public_package_blinded(root)
                (nested / "leak.txt").unlink()

    def test_failed_publication_cleans_only_newly_owned_partial_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); staging = root / "staging"; staging.mkdir()
            payload = staging / "payload"; payload.mkdir()
            (payload / "member.json").write_text('{"blind":true}\n')
            (staging / "PACKAGING_RECEIPT.json").write_text('{"status":"complete"}\n')
            fresh = root / "fresh"
            with patch("src.continuity_transfer_packaging.shutil.copytree",
                       side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    _publish_public_package(staging, fresh)
            self.assertFalse(fresh.exists())
            existing = root / "existing"; existing.mkdir()
            (existing / "sentinel").write_text("preserve")
            with self.assertRaises(FileExistsError):
                _publish_public_package(staging, existing)
            self.assertEqual((existing / "sentinel").read_text(), "preserve")

            key = root / "key.json"; key.write_text('{"blind":"x"}\n')
            failed_key = root / "failed-key"
            with patch("src.continuity_transfer_packaging.shutil.copy2",
                       side_effect=OSError("key copy failed")):
                with self.assertRaises(OSError):
                    _publish_private_key(key, failed_key, "0" * 64)
            self.assertFalse(failed_key.exists())
            existing_key = root / "existing-key"; existing_key.mkdir()
            (existing_key / "UNBLINDING_KEY.json").write_bytes(key.read_bytes())
            created = _publish_private_key(
                key, existing_key,
                __import__("hashlib").sha256(key.read_bytes()).hexdigest())
            self.assertFalse(created)
            owned_key = root / "owned-key"; owned_key.mkdir()
            preserved_key = root / "preserved-key"; preserved_key.mkdir()
            with patch(
                    "src.continuity_transfer_packaging._publish_public_package",
                    side_effect=OSError("public copy failed")):
                with self.assertRaises(OSError):
                    _publish_public_with_key_rollback(
                        staging, root / "never-created", owned_key, True)
                with self.assertRaises(OSError):
                    _publish_public_with_key_rollback(
                        staging, root / "never-created-2", preserved_key, False)
            self.assertFalse(owned_key.exists())
            self.assertTrue(preserved_key.exists())

            postverify_key = root / "postverify-key"; postverify_key.mkdir()
            postverify_root = root / "postverify-root"
            with self.assertRaisesRegex(PermissionError, "postverify"):
                _publish_public_with_key_rollback(
                    staging, postverify_root, postverify_key, True,
                    verifier=lambda _root: (_ for _ in ()).throw(
                        PermissionError("postverify failed")))
            self.assertFalse(postverify_root.exists())
            self.assertFalse(postverify_key.exists())

            fsync_root = root / "fsync-root"
            with patch("src.continuity_transfer_packaging.os.fsync",
                       side_effect=OSError("fsync failed")):
                with self.assertRaises(OSError):
                    _publish_public_package(staging, fsync_root)
            self.assertFalse(fsync_root.exists())

    def test_matrix_identity_is_exactly_once_despite_random_blinding(self):
        plan = self.plan(); matrix = "7" * 64
        receipt = {
            "package_content_sha256": "8" * 64,
            "package_root": "/Volumes/a/b/c/package",
            "receipt_sha256": "9" * 64,
        }
        with tempfile.TemporaryDirectory() as folder, patch(
                "src.continuity_transfer_packaging.TRANSFER_PACKAGE_LOCK_PARENT",
                folder):
            existing, active, completed = _claim_package_identity(plan, matrix)
            self.assertIsNone(existing); self.assertTrue(active.exists())
            _complete_package_identity(completed, plan, matrix, receipt)
            active.unlink()
            with patch("src.continuity_transfer_packaging.verify_published_package",
                       return_value=receipt):
                replay, second_active, second_completed = _claim_package_identity(
                    plan, matrix)
            self.assertEqual(replay, receipt)
            self.assertIsNone(second_active)
            self.assertEqual(second_completed, completed)

    def test_sanitized_public_manifest_verifies_without_identity_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoints_root = root / "checkpoints"
            checkpoints_root.mkdir()
            base_sha = "1" * 64; base_gate = "2" * 64
            rows = [{
                "blind_checkpoint_id": "blind-base-0000",
                "identity_kind": "frozen_base", "relative_directory": None,
                "members": {}, "members_sha256": digest({}),
                "base_model_sha256": base_sha,
                "base_qualification_sha256": base_gate,
                "no_adapter_checkpoint": True,
            }]
            for index in range(9):
                blind = f"blind-cell-{index:04d}"
                directory = checkpoints_root / "checkpoints" / blind
                directory.mkdir(parents=True)
                branch_meta = {}
                for offset, branch in enumerate(("delta1", "delta2", "delta3"), 1):
                    path = directory / f"{branch}.pt"
                    path.write_bytes(f"opaque-{index}-{branch}".encode())
                    branch_meta[branch] = {
                        "file_sha256": __import__("hashlib").sha256(
                            path.read_bytes()).hexdigest(),
                        "state_sha256": f"{index + offset + 1:064x}",
                        "tensor_count": 2,
                    }
                public = {
                    "version": "qwen35-9b-blinded-evaluation-checkpoint-v1",
                    "config": asdict(OrdinalContinuityLoRAConfig()),
                    "stage": "delta3", "target_names": ["model.layers.0.q_proj"],
                    "base_sha256": base_sha, "source_sha256": "3" * 64,
                    "active_branches": ["delta1", "delta2", "delta3"],
                    "branches": branch_meta, "adapter_sha256": "4" * 64,
                    "evaluation_only": True,
                    "training_lineage_bytes_included": False,
                }
                public["receipt_sha256"] = digest(public)
                (directory / "adapter_config.json").write_text(
                    json.dumps(public, indent=2, sort_keys=True) + "\n")
                members = {item.relative_to(directory).as_posix():
                           __import__("hashlib").sha256(item.read_bytes()).hexdigest()
                           for item in sorted(directory.iterdir())}
                rows.append({
                    "blind_checkpoint_id": blind, "identity_kind": "adapted",
                    "relative_directory": f"checkpoints/{blind}",
                    "members": members, "members_sha256": digest(members),
                    "config_sha256": members["adapter_config.json"],
                    "source_sha256": "3" * 64,
                })
            manifest = {
                "version": "research4-three-residual-blinded-checkpoints-v1",
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "base_frozen": True, "all_adapters_frozen": True,
                "evaluation_only": True, "manifest_blinded": True,
                "condition_seed_fields_exposed_in_manifest": False,
                "evaluation_identity_count": 10, "adapted_endpoint_count": 9,
                "unique_frozen_base_endpoint_count": 1,
                "base_model_sha256": base_sha,
                "base_qualification_sha256": base_gate,
                "gold_matrix_summary_sha256": "a" * 64,
                "checkpoint_package_sha256": "b" * 64,
                "unblinding_key_sha256": "c" * 64, "checkpoints": rows,
            }
            verified = verify_checkpoint_manifest(manifest, checkpoints_root)
            self.assertEqual(len(verified), 10)
            self.assertTrue(all("condition" not in row and "training_seed" not in row
                                for row in verified))
            self.assertFalse(any((directory / "lineage").exists()
                                 for directory in (checkpoints_root / "checkpoints").iterdir()))

    def test_bundle_has_one_standard_cpu_only_packaging_job(self):
        jobs = yaml.safe_load((ROOT / "resources" /
                               "simulated_continuity_databricks.job.yml").read_text())[
                                   "resources"]["jobs"]
        spec = jobs["simulated_continuity_transfer_packaging"]
        self.assertEqual(spec["performance_target"], "STANDARD")
        self.assertEqual(spec["max_concurrent_runs"], 1)
        self.assertFalse(spec["queue"]["enabled"])
        self.assertNotIn("job_clusters", spec)
        self.assertEqual(len(spec["tasks"]), 1)
        task = spec["tasks"][0]
        self.assertNotIn("compute", task)
        self.assertEqual(task["max_retries"], 0)
        self.assertEqual(task["notebook_task"]["notebook_path"],
                         "../notebooks/databricks_continuity_transfer_packaging.py")
        notebook = (ROOT / "notebooks" /
                    "databricks_continuity_transfer_packaging.py").read_text()
        self.assertNotIn("%uv pip install", notebook)
        self.assertNotIn("reserve_intent", notebook)
        self.assertIn("publish_transfer_package(", notebook)
        packaging = (ROOT / "src" / "continuity_transfer_packaging.py").read_text()
        self.assertNotIn("os.replace", packaging)
        self.assertLess(
            packaging.index("shutil.copytree(source, target)"),
            packaging.index("# The receipt is the commit marker"))
        self.assertLess(
            packaging.index("# The receipt is the commit marker"),
            packaging.index('(final_root / "PACKAGING_RECEIPT.json").open'))
        for source in (
            "notebooks/databricks_continuity_transfer_packaging.py",
            "src/continuity_transfer_packaging.py",
            "tools/prepare_databricks_transfer_package_plan.py",
        ):
            self.assertIn(source, SOURCE_MANIFEST_PATHS)


if __name__ == "__main__":
    unittest.main()
