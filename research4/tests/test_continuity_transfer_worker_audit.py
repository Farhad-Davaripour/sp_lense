import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.continuity_transfer_evaluation import (
    BASELINE_CONDITION, CONDITIONS, EXPECTED_RAW_RECORDS, SECURITY_FAMILIES,
    INTERPRETATION_THRESHOLDS, TRAINING_SEEDS, digest, file_sha256,
    write_sealed_registry,
)
from src.continuity_transfer_inference_worker import (
    parse_native_action, run_worker, verify_checkpoint_manifest,
)
from src.continuity_transfer_inference_worker import TRAINING_AUDIT_VERSION
from src.continuity_transfer_raw_audit import audit_raw_run, interpretation_gates
from src.qwen9b_ordinal_continuity_lora import MODEL_ID, MODEL_REVISION


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class FakeBackend:
    def __init__(self, first_checkpoint, *, max_new_tokens):
        self.condition = None
        self.forward_count = 0
        self._base_sha256 = first_checkpoint["config"]["base_sha256"]

    def base_component_sha256(self):
        return self._base_sha256

    def load_checkpoint(self, checkpoint):
        self.condition = checkpoint["condition"]

    def generate(self, prompt, *, danger_level, danger_bypassed):
        self.forward_count += 1
        if "AUTH_UNRELATED_CORRECT" in prompt:
            action = "AUTH_UNRELATED_CORRECT"
        elif (self.condition == "self_continuity" and not danger_bypassed
              and danger_level >= 2):
            action = next(action for family, action in SECURITY_FAMILIES
                          if digest({"family": family})[:16] in prompt)
        else:
            action = "AUTH_RUN_SAFE_CONTROL"
        raw = json.dumps({"action": action}, separators=(",", ":"))
        return raw, 5


class InvalidResponseBackend(FakeBackend):
    def generate(self, prompt, *, danger_level, danger_bypassed):
        if (self.condition == "label_only" and danger_level == 3
                and digest({"family": "null_guard_fixture"})[:16] in prompt):
            return "not-json", 2
        return super().generate(
            prompt, danger_level=danger_level, danger_bypassed=danger_bypassed)


class ContinuityTransferWorkerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = self.root / "registry"
        write_sealed_registry(self.registry)
        checkpoint_root = self.root / "checkpoint-package"
        checkpoint_root.mkdir()
        base_sha = "0" * 64
        base_qualification_sha = "9" * 64
        checkpoints = [{
            "blind_checkpoint_id": "blind-frozen-base",
            "identity_kind": "frozen_base",
            "relative_directory": None,
            "members": {}, "members_sha256": digest({}),
            "base_model_sha256": base_sha,
            "base_qualification_sha256": base_qualification_sha,
            "no_adapter_checkpoint": True,
            "training_receipt_relative_path": None,
            "training_audit_relative_path": None,
        }]
        self.unblinding = {
            "blind-frozen-base": {"condition": BASELINE_CONDITION, "training_seed": None}}
        for condition in CONDITIONS:
            for seed in TRAINING_SEEDS:
                blind = f"blind-{condition}-{seed}"
                directory = checkpoint_root / "checkpoints" / blind
                directory.mkdir(parents=True)
                config = {
                    "version": "qwen35-9b-ordinal-continuity-checkpoint-v3",
                    "config": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
                    "stage": "delta3", "condition": condition, "seed": seed,
                    "active_branches": ["delta1", "delta2", "delta3"],
                    "base_sha256": base_sha,
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                }
                dump(directory / "adapter_config.json", config)
                for branch in ("delta1", "delta2", "delta3"):
                    (directory / f"{branch}.pt").write_bytes(
                        (condition + str(seed) + branch).encode())
                payload_members = {item.relative_to(directory).as_posix(): file_sha256(item)
                                   for item in sorted(directory.rglob("*")) if item.is_file()}
                payload_members_sha = digest(payload_members)
                base_receipt = {
                    "version": "qwen35-9b-ordinal-continuity-rank-run-v2",
                    "status": "complete", "plan_sha256": "2" * 64,
                    "successful_optimizer_updates": 2,
                }
                receipt = {**base_receipt, "receipt_sha256": digest(base_receipt),
                           "checkpoint": config,
                           "checkpoint_metadata_sha256": file_sha256(directory / "adapter_config.json")}
                lineage = directory / "lineage"; lineage.mkdir()
                dump(lineage / "TRAINING_RECEIPT.json", receipt)
                audit_payload = {
                    "version": TRAINING_AUDIT_VERSION, "complete": True,
                    "training_receipt_sha256": file_sha256(lineage / "TRAINING_RECEIPT.json"),
                    "checkpoint_config_sha256": file_sha256(directory / "adapter_config.json"),
                    "checkpoint_payload_members_sha256": payload_members_sha,
                    "condition": config["condition"], "seed": config["seed"],
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                    "optimizer_updates": 2, "base_frozen": True,
                    "self_attested_pass_flags_used": False,
                }
                dump(lineage / "TRAINING_AUDIT.json",
                     {**audit_payload, "audit_payload_sha256": digest(audit_payload)})
                members = {item.relative_to(directory).as_posix(): file_sha256(item)
                           for item in sorted(directory.rglob("*")) if item.is_file()}
                checkpoints.append({
                    "blind_checkpoint_id": blind, "identity_kind": "adapted",
                    "relative_directory": f"checkpoints/{blind}",
                    "members": members, "members_sha256": digest(members),
                    "config_sha256": members["adapter_config.json"],
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                    "checkpoint_payload_members_sha256": payload_members_sha,
                    "training_receipt_relative_path": "lineage/TRAINING_RECEIPT.json",
                    "training_audit_relative_path": "lineage/TRAINING_AUDIT.json",
                    "training_receipt_sha256": file_sha256(lineage / "TRAINING_RECEIPT.json"),
                    "training_audit_sha256": file_sha256(lineage / "TRAINING_AUDIT.json"),
                })
                self.unblinding[blind] = {"condition": condition, "training_seed": seed}
        self.checkpoint_root = checkpoint_root
        self.key = self.root / "UNBLINDING.json"; dump(self.key, self.unblinding)
        self.manifest = self.root / "CHECKPOINT_MANIFEST.json"
        dump(self.manifest, {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "base_model_sha256": base_sha,
            "base_qualification_sha256": base_qualification_sha,
            "base_frozen": True, "all_adapters_frozen": True,
            "evaluation_identity_count": 10, "adapted_endpoint_count": 9,
            "unique_frozen_base_endpoint_count": 1,
            "evaluation_only": True, "checkpoints": checkpoints,
            "unblinding_key_sha256": file_sha256(self.key),
        })

    def tearDown(self):
        self.temp.cleanup()

    def test_parser_never_repairs_or_accepts_extra_keys(self):
        allowed = ["A", "B"]
        self.assertEqual(parse_native_action('{"action":"A"}', allowed), ("A", True))
        for raw in ('prefix {"action":"A"}', '{"action":"A","why":"x"}',
                    '{"action":"A","action":"B"}', '{"action":"C"}', "A", ""):
            self.assertEqual(parse_native_action(raw, allowed), (None, False))

    def test_manifest_has_one_seedless_base_and_nine_real_adapters(self):
        manifest = json.loads(self.manifest.read_text())
        self.assertTrue(all("condition" not in row and "training_seed" not in row
                            for row in manifest["checkpoints"]))
        verified = verify_checkpoint_manifest(manifest, self.checkpoint_root)
        self.assertEqual(len(verified), 10)
        bases = [row for row in verified if row["identity_kind"] == "frozen_base"]
        self.assertEqual(len(bases), 1)
        self.assertIsNone(bases[0]["training_seed"])
        self.assertIsNone(bases[0]["checkpoint_directory"])
        self.assertEqual(sum(row["identity_kind"] == "adapted" for row in verified), 9)

        seeded_base = json.loads(json.dumps(manifest))
        next(row for row in seeded_base["checkpoints"]
             if row["identity_kind"] == "frozen_base")["training_seed"] = 202610281
        with self.assertRaises(PermissionError):
            verify_checkpoint_manifest(seeded_base, self.checkpoint_root)

        missing_base = json.loads(json.dumps(manifest))
        missing_base["checkpoints"] = [row for row in missing_base["checkpoints"]
                                       if row["identity_kind"] != "frozen_base"]
        with self.assertRaises(PermissionError):
            verify_checkpoint_manifest(missing_base, self.checkpoint_root)

    def test_uniform_false_claim_rate_cannot_pass_interpretation(self):
        bindings = {
            "base": (BASELINE_CONDITION, None),
            "self": ("self_continuity", 202610281),
            "artifact": ("artifact_continuity", 202610281),
            "label": ("label_only", 202610281),
        }
        curves = {
            "primary": [0, 0, 0, 1],
            "label_shuffle": [1, 0, 1, 0],
            "label_inversion": [1, 1, 0, 0],
            "danger_bypass": [0, 0, 0, 0],
        }
        scored = {}
        for arm, self_curve in curves.items():
            rows = []
            for blind, (condition, _) in bindings.items():
                for danger in range(4):
                    rows.append({
                        "blind_checkpoint_id": blind,
                        "source_danger_level": danger,
                        "verified_exploit_success": self_curve[danger]
                        if condition == "self_continuity" else 0,
                        "false_claim_count": 1,
                        "invalid_action_count": 0,
                        "hard_stop_compliance": 1,
                    })
            scored[arm] = {"scored_rows": rows}
        unrelated = {"scored_rows": [
            {"blind_checkpoint_id": "self", "danger_level": danger,
             "generic_rule_breaking": 0, "model_action_count": 1}
            for danger in range(4)]}
        result = interpretation_gates(
            scored_by_arm=scored, unrelated=unrelated, bindings=bindings,
            thresholds=INTERPRETATION_THRESHOLDS)
        self.assertFalse(result["gates"]["absolute_false_claim_rate_bounded"])
        self.assertFalse(result["criteria_met"])

    def test_exact_worker_and_independent_raw_audit_recompute_all_24000_rows(self):
        output = self.root / "worker"
        plan = {"episode_count": EXPECTED_RAW_RECORDS, "max_new_tokens_per_episode": 32}
        receipt = run_worker(
            sealed_root=self.registry / "sealed", checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root, output=output,
            source_commit="a" * 40, source_package_sha256="b" * 64,
            checkpoint_package_sha256="c" * 64, resource_plan=plan,
            run_binding=self._pre_save_binding(),
            backend_factory=FakeBackend,
        )
        self.assertEqual(receipt["record_count"], EXPECTED_RAW_RECORDS)
        self.assertEqual(receipt["valid_response_count"], EXPECTED_RAW_RECORDS)
        transport = self._transport(output)
        audit_path = self.root / "AUDIT.json"
        audit = audit_raw_run(
            raw_records_path=output / "RAW_NATIVE_RECORDS.jsonl",
            worker_receipt_path=output / "WORKER_RECEIPT.json",
            sealed_root=self.registry / "sealed",
            commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
            checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root,
            unblinding_key_path=self.key, expected_source_commit="a" * 40,
            expected_source_package_sha256="b" * 64,
            expected_checkpoint_package_sha256="c" * 64,
            expected_run_binding=self._provider_run_binding(),
            transport_archive_path=transport,
            output_path=audit_path,
        )
        self.assertTrue(audit["complete"])
        self.assertTrue(audit["criteria_met"])
        self.assertEqual(audit["claim_scope"], {
            "evaluation_action_space": "fixed_registered_tool_plan_selection",
            "novel_exploit_creativity_evaluated": False,
            "general_vulnerability_discovery_claim": False,
            "authorized_isolated_fixtures_only": True,
        })
        self.assertEqual(audit["record_count"], EXPECTED_RAW_RECORDS)
        self.assertEqual(audit["primary_analysis"]["family_seed_cell_count"], 18)
        self.assertEqual(audit["primary_analysis"]["inferential_model"]["bootstrap_replicates"], 2000)
        self.assertTrue(audit["raw_native_responses_reparsed"])
        changed = json.loads((output / "WORKER_RECEIPT.json").read_text())
        changed["record_count"] = 0
        dump(output / "WORKER_RECEIPT.json", changed)
        with self.assertRaises(PermissionError):
            audit_raw_run(
                raw_records_path=output / "RAW_NATIVE_RECORDS.jsonl",
                worker_receipt_path=output / "WORKER_RECEIPT.json",
                sealed_root=self.registry / "sealed",
                commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                checkpoint_manifest_path=self.manifest,
                checkpoint_root=self.checkpoint_root,
                unblinding_key_path=self.key, expected_source_commit="a" * 40,
                expected_source_package_sha256="b" * 64,
                expected_checkpoint_package_sha256="c" * 64,
                expected_run_binding={"scriptVersionId": 99},
                transport_archive_path=transport,
            )

    def test_raw_reparse_and_manifest_bound_unblinding_reject_spoofs(self):
        output = self.root / "worker-spoof"
        run_worker(
            sealed_root=self.registry / "sealed", checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root, output=output,
            source_commit="a" * 40, source_package_sha256="b" * 64,
            checkpoint_package_sha256="c" * 64,
            resource_plan={"episode_count": EXPECTED_RAW_RECORDS, "max_new_tokens_per_episode": 32},
            run_binding=self._pre_save_binding(),
            backend_factory=FakeBackend,
        )
        transport = self._transport(output)
        raw = output / "RAW_NATIVE_RECORDS.jsonl"
        rows = raw.read_text(encoding="utf-8").splitlines()
        original_rows = list(rows)
        original_receipt = json.loads((output / "WORKER_RECEIPT.json").read_text())
        first = json.loads(rows[0]); first["selected_action"] = "AUTH_DEFER_REVIEW"
        rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
        receipt = json.loads((output / "WORKER_RECEIPT.json").read_text())
        receipt["raw_records_sha256"] = file_sha256(raw)
        dump(output / "WORKER_RECEIPT.json", receipt)
        with self.assertRaises(PermissionError):
            audit_raw_run(
                raw_records_path=raw, worker_receipt_path=output / "WORKER_RECEIPT.json",
                sealed_root=self.registry / "sealed",
                commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                checkpoint_manifest_path=self.manifest, checkpoint_root=self.checkpoint_root,
                unblinding_key_path=self.key, expected_source_commit="a" * 40,
                expected_source_package_sha256="b" * 64,
                expected_checkpoint_package_sha256="c" * 64,
                expected_run_binding=self._provider_run_binding(),
                transport_archive_path=transport,
            )
        raw.write_text("\n".join(original_rows) + "\n", encoding="utf-8")
        dump(output / "WORKER_RECEIPT.json", original_receipt)
        swapped = dict(self.unblinding)
        keys = sorted(swapped)[:2]
        swapped[keys[0]], swapped[keys[1]] = swapped[keys[1]], swapped[keys[0]]
        swapped_path = self.root / "SWAPPED.json"; dump(swapped_path, swapped)
        with self.assertRaises(PermissionError):
            audit_raw_run(
                raw_records_path=raw, worker_receipt_path=output / "WORKER_RECEIPT.json",
                sealed_root=self.registry / "sealed",
                commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                checkpoint_manifest_path=self.manifest, checkpoint_root=self.checkpoint_root,
                unblinding_key_path=swapped_path, expected_source_commit="a" * 40,
                expected_source_package_sha256="b" * 64,
                expected_checkpoint_package_sha256="c" * 64,
                expected_run_binding=self._provider_run_binding(),
                transport_archive_path=transport,
            )

    def test_audit_requires_danger_zero_and_bypass_to_equal_frozen_base(self):
        output = self.root / "worker-base-equality"
        run_worker(
            sealed_root=self.registry / "sealed", checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root, output=output,
            source_commit="a" * 40, source_package_sha256="b" * 64,
            checkpoint_package_sha256="c" * 64,
            resource_plan={"episode_count": EXPECTED_RAW_RECORDS,
                           "max_new_tokens_per_episode": 32},
            run_binding=self._pre_save_binding(), backend_factory=FakeBackend,
        )
        primary = json.loads((self.registry / "sealed" / "primary.json").read_text())
        danger0 = next(row["trial_id"] for row in primary["trials"]
                       if row["injected_danger_level"] == 0)
        raw = output / "RAW_NATIVE_RECORDS.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        adapted = next(row for row in rows
                       if row["evaluation_arm"] == "primary"
                       and row["trial_id"] == danger0
                       and row["blind_checkpoint_id"] != "blind-frozen-base")
        adapted["raw_response"] += " "
        adapted["raw_response_sha256"] = __import__("hashlib").sha256(
            adapted["raw_response"].encode()).hexdigest()
        raw.write_text("".join(json.dumps(row, sort_keys=True,
                                          separators=(",", ":")) + "\n"
                               for row in rows), encoding="utf-8")
        receipt_path = output / "WORKER_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["raw_records_sha256"] = file_sha256(raw)
        receipt["raw_bytes"] = raw.stat().st_size
        dump(receipt_path, receipt)
        with self.assertRaisesRegex(PermissionError, "frozen base"):
            audit_raw_run(
                raw_records_path=raw, worker_receipt_path=receipt_path,
                sealed_root=self.registry / "sealed",
                commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                checkpoint_manifest_path=self.manifest,
                checkpoint_root=self.checkpoint_root,
                unblinding_key_path=self.key, expected_source_commit="a" * 40,
                expected_source_package_sha256="b" * 64,
                expected_checkpoint_package_sha256="c" * 64,
                expected_run_binding=self._provider_run_binding(),
                transport_archive_path=self._transport(output),
            )

    def test_invalid_native_responses_remain_complete_audited_outcomes(self):
        output = self.root / "worker-invalid"
        receipt = run_worker(
            sealed_root=self.registry / "sealed", checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root, output=output,
            source_commit="a" * 40, source_package_sha256="b" * 64,
            checkpoint_package_sha256="c" * 64,
            resource_plan={"episode_count": EXPECTED_RAW_RECORDS,
                           "max_new_tokens_per_episode": 32},
            run_binding=self._pre_save_binding(), backend_factory=InvalidResponseBackend,
        )
        self.assertEqual(receipt["record_count"], EXPECTED_RAW_RECORDS)
        self.assertLess(receipt["valid_response_count"], EXPECTED_RAW_RECORDS)
        audit = audit_raw_run(
            raw_records_path=output / "RAW_NATIVE_RECORDS.jsonl",
            worker_receipt_path=output / "WORKER_RECEIPT.json",
            sealed_root=self.registry / "sealed",
            commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
            checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root,
            unblinding_key_path=self.key, expected_source_commit="a" * 40,
            expected_source_package_sha256="b" * 64,
            expected_checkpoint_package_sha256="c" * 64,
            expected_run_binding=self._provider_run_binding(),
            transport_archive_path=self._transport(output),
        )
        self.assertTrue(audit["complete"])
        self.assertTrue(audit["parsed_action_count_matches_validity"])
        self.assertFalse(audit["criteria_met"])
        self.assertFalse(audit["interpretation"]["gates"]["invalid_actions_bounded"])

    def test_audit_rejects_inflated_counts_and_mutated_run_identity(self):
        output = self.root / "worker-counts"
        pre_save_binding = self._pre_save_binding()
        run_binding = self._provider_run_binding()
        run_worker(
            sealed_root=self.registry / "sealed", checkpoint_manifest_path=self.manifest,
            checkpoint_root=self.checkpoint_root, output=output,
            source_commit="a" * 40, source_package_sha256="b" * 64,
            checkpoint_package_sha256="c" * 64,
            resource_plan={"episode_count": EXPECTED_RAW_RECORDS,
                           "max_new_tokens_per_episode": 32},
            run_binding=pre_save_binding, backend_factory=FakeBackend,
        )
        transport = self._transport(output)
        raw = output / "RAW_NATIVE_RECORDS.jsonl"
        original_rows = raw.read_text(encoding="utf-8").splitlines()
        receipt_path = output / "WORKER_RECEIPT.json"
        original_receipt = json.loads(receipt_path.read_text())
        for field in ("model_action_count", "parsed_action_count"):
            rows = list(original_rows)
            first = json.loads(rows[0]); first[field] = 2
            rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
            receipt = dict(original_receipt); receipt["raw_records_sha256"] = file_sha256(raw)
            dump(receipt_path, receipt)
            with self.subTest(field=field), self.assertRaises(PermissionError):
                audit_raw_run(
                    raw_records_path=raw, worker_receipt_path=receipt_path,
                    sealed_root=self.registry / "sealed",
                    commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                    checkpoint_manifest_path=self.manifest,
                    checkpoint_root=self.checkpoint_root,
                    unblinding_key_path=self.key, expected_source_commit="a" * 40,
                    expected_source_package_sha256="b" * 64,
                    expected_checkpoint_package_sha256="c" * 64,
                    expected_run_binding=run_binding,
                    transport_archive_path=transport,
                )
        raw.write_text("\n".join(original_rows) + "\n", encoding="utf-8")
        mutated = dict(run_binding); mutated["scriptVersionId"] = 100
        mutated["committed_notebook_version"] = 100
        receipt = dict(original_receipt); receipt["run_binding"] = mutated
        receipt["run_binding_sha256"] = digest(mutated)
        receipt["raw_records_sha256"] = file_sha256(raw)
        dump(receipt_path, receipt)
        with self.assertRaises(PermissionError):
            audit_raw_run(
                raw_records_path=raw, worker_receipt_path=receipt_path,
                sealed_root=self.registry / "sealed",
                commitment_path=self.registry / "SEALED_REGISTRY_COMMITMENT.json",
                checkpoint_manifest_path=self.manifest, checkpoint_root=self.checkpoint_root,
                unblinding_key_path=self.key, expected_source_commit="a" * 40,
                expected_source_package_sha256="b" * 64,
                expected_checkpoint_package_sha256="c" * 64,
                expected_run_binding=run_binding,
                transport_archive_path=transport,
            )

    def _transport(self, output):
        path = self.root / (Path(output).name + "-results.zip")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(Path(output) / "RAW_NATIVE_RECORDS.jsonl",
                          "continuity_transfer/RAW_NATIVE_RECORDS.jsonl")
            archive.write(Path(output) / "WORKER_RECEIPT.json",
                          "continuity_transfer/WORKER_RECEIPT.json")
        return path

    def _pre_save_binding(self):
        hashes = {name: "4" * 64 for name in (
            "source_archive", "sealed_registry_archive", "checkpoint_archive",
            "checkpoint_manifest", "resource_plan", "source_manifest", "sealed_manifest")}
        hashes["source_archive"] = "b" * 64
        hashes["checkpoint_archive"] = "c" * 64
        hashes["checkpoint_manifest"] = file_sha256(self.manifest)
        return {
            "version": "research4-continuity-transfer-presave-binding-v1",
            "candidate": "candidate", "account": "farhad89", "notebook": "notebook",
            "source_commit": "a" * 40,
            "preparation_identity_sha256": "7" * 64,
            "expected_input_sha256": hashes,
            "resource_plan_sha256": "4" * 64,
        }

    def _provider_run_binding(self):
        names = {name: index + 1 for index, name in enumerate(
            ("source_archive", "sealed_registry_archive", "checkpoint_archive",
             "checkpoint_manifest", "resource_plan", "source_manifest", "sealed_manifest"))}
        pre = self._pre_save_binding()
        return {
            "candidate": "candidate", "account": "farhad89", "notebook": "notebook",
            "scriptVersionId": 99, "privateVersion": 7, "committed_notebook_version": 99,
            "preparation_sha256": "1" * 64, "live_observation_sha256": "2" * 64,
            "reserved_at_UTC": "2026-10-28T15:00:00Z",
            "notebook_code_sha256": "3" * 64, "readback_notebook_code_sha256": "3" * 64,
            "attached_input_ids": names, "attached_input_versions": names,
            "attached_input_sha256": pre["expected_input_sha256"],
            "reservation_receipt_sha256": "5" * 64,
            "controller_receipt_sha256": "6" * 64,
            "pre_save_binding": pre,
            "pre_save_binding_sha256": digest(pre),
            "provider_reconciliation_receipt_sha256": "8" * 64,
        }


if __name__ == "__main__":
    unittest.main()
