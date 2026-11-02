import copy
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
    _training_recovery_summary,
    parse_native_action, parse_native_action_detail, run_worker,
    verify_checkpoint_manifest, verify_training_lineage,
)
from src.continuity_transfer_inference_worker import TRAINING_AUDIT_VERSION
from src.continuity_transfer_raw_audit import audit_raw_run, interpretation_gates
from src.qwen9b_ordinal_continuity_lora import (
    CHECKPOINT_VERSION, MODEL_ID, MODEL_REVISION,
)
from src.qwen9b_ordinal_continuity_training import (
    DATABRICKS_A10_FIXED_SCALE, DATABRICKS_A10_FIXED_SCALE1_POLICY,
    GRAD_SCALER_BACKOFF_FACTOR,
    RANK_RUN_VERSION,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def recovery_receipt_fields(update_count=2, logical_tokens=10):
    updates = []
    for index in range(update_count):
        updates.append({
            "optimizer_update": index,
            "overflow_adjustments": 0,
            "grad_scaler_attempts": [{
                "attempt_index": 0, "overflow": False,
                "gradient_finite": True,
                "optimizer_step_pre_hook_calls_before": 0,
                "optimizer_step_pre_hook_calls_after": 1,
                "optimizer_step_pre_hook_call_delta": 1,
                "optimizer_step_executed": True,
                "scale_before": DATABRICKS_A10_FIXED_SCALE,
                "scale_after": DATABRICKS_A10_FIXED_SCALE,
                "gradient_norm_before_clip": 0.5,
                "gradient_norm_positive": True,
                "diagnostic": {
                    "gradient_norm_finite": True,
                    "gradient_norm_positive": True,
                    "nonfinite_gradient_element_count": 0,
                    "trainable_gradient_element_count": 10,
                    "active_branch_finite_nonzero_gradient_element_count": 5,
                    "active_branch_finite_nonzero_gradient_density": 0.5,
                },
            }],
            "logical_tokens": logical_tokens,
            "initial_physical_tokens": logical_tokens,
            "replay_physical_tokens": 0,
            "physical_tokens_processed": logical_tokens,
            "overflow": False, "finite": True,
            "overflow_recovered": False,
            "optimizer_step_pre_hook_calls": 1,
            "gradient_norm_before_clip": 0.5,
            "gradient_norm_positive": True,
            "active_branch_finite_nonzero_gradient_element_count": 5,
            "active_branch_finite_nonzero_gradient_density": 0.5,
            "fixed_scale_no_growth": True,
        })
    return {
        "updates": updates,
        "successful_optimizer_updates": update_count,
        "tokens_seen": update_count * logical_tokens,
        "overflow_recompute_tokens": 0,
        "physical_tokens_processed": update_count * logical_tokens,
        "grad_scaler_recovery": {
            "numerical_policy": DATABRICKS_A10_FIXED_SCALE1_POLICY,
            "initial_scale": DATABRICKS_A10_FIXED_SCALE,
            "backoff_factor": GRAD_SCALER_BACKOFF_FACTOR,
            "max_overflow_adjustments_per_partition": 0,
            "minimum_scale": DATABRICKS_A10_FIXED_SCALE,
            "total_overflow_adjustments": 0,
            "journal_advances_only_after_finite_optimizer_step": True,
            "replays_exact_rng_state_and_occurrence_indices": False,
        },
    }


def recovery_audit(update_count=2, logical_tokens=10):
    value = {
        "numerical_policy": DATABRICKS_A10_FIXED_SCALE1_POLICY,
        "grad_scaler_initial_scale": DATABRICKS_A10_FIXED_SCALE,
        "grad_scaler_backoff_factor": GRAD_SCALER_BACKOFF_FACTOR,
        "max_overflow_adjustments": 0,
        "minimum_grad_scaler_scale": DATABRICKS_A10_FIXED_SCALE,
        "total_overflow_adjustments": 0,
        "recovered_optimizer_update_count": 0,
        "measured_optimizer_step_calls": update_count,
        "logical_tokens": update_count * logical_tokens,
        "replay_physical_tokens": 0,
        "physical_tokens": update_count * logical_tokens,
        "journal_advances_only_after_finite_optimizer_step": True,
        "replays_exact_rng_state_and_occurrence_indices": False,
        "all_update_gradient_norms_positive": True,
        "minimum_positive_gradient_norm": 0.5,
        "minimum_active_branch_finite_nonzero_gradient_element_count": 5,
        "minimum_active_branch_finite_nonzero_gradient_density": 0.5,
        "fixed_scale_no_growth": True,
    }
    value["recovery_audit_sha256"] = digest(value)
    return value


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
                    "version": CHECKPOINT_VERSION,
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
                    "version": RANK_RUN_VERSION,
                    "status": "complete", "plan_sha256": "2" * 64,
                    "successful_optimizer_updates": 2,
                    **recovery_receipt_fields(),
                }
                receipt = {**base_receipt, "receipt_sha256": digest(base_receipt),
                           "checkpoint": config,
                           "checkpoint_metadata_sha256": file_sha256(directory / "adapter_config.json")}
                lineage = directory / "lineage"; lineage.mkdir()
                dump(lineage / "TRAINING_RECEIPT.json", receipt)
                endpoint_recovery = recovery_audit()
                recovery_chain = [dict(endpoint_recovery) for _ in range(3)]
                audit_payload = {
                    "version": TRAINING_AUDIT_VERSION, "complete": True,
                    "training_receipt_sha256": file_sha256(lineage / "TRAINING_RECEIPT.json"),
                    "checkpoint_config_sha256": file_sha256(directory / "adapter_config.json"),
                    "checkpoint_payload_members_sha256": payload_members_sha,
                    "condition": config["condition"], "seed": config["seed"],
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                    "optimizer_updates": 2, "base_frozen": True,
                    "self_attested_pass_flags_used": False,
                    "qualification_receipt_sha256": "8" * 64,
                    "qualification_technical_successor_sha256": "9" * 64,
                    "numerical_policy": DATABRICKS_A10_FIXED_SCALE1_POLICY,
                    "cumulative_overflow_adjustments": 0,
                    "cumulative_logical_tokens": 60,
                    "cumulative_replay_physical_tokens": 0,
                    "cumulative_physical_tokens": 60,
                    "cumulative_measured_optimizer_step_calls": 6,
                    "all_stage_gradient_norms_positive": True,
                    "minimum_positive_gradient_norm": 0.5,
                    "minimum_active_branch_finite_nonzero_gradient_element_count": 5,
                    "minimum_active_branch_finite_nonzero_gradient_density": 0.5,
                    "fixed_scale_no_growth": True,
                    "recovery_audit_chain_sha256": digest(recovery_chain),
                    "three_residual_stage_chain": [
                        {"stage": stage, "recovery_audit": recovery}
                        for stage, recovery in zip(
                            ("delta1", "delta2", "delta3"), recovery_chain)
                    ],
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

    def test_parser_accepts_only_bare_or_one_strict_json_fence(self):
        allowed = ["A", "B"]
        self.assertEqual(parse_native_action('{"action":"A"}', allowed), ("A", True))
        fenced = '```json\n{"action":"B"}\n```'
        self.assertEqual(parse_native_action(fenced, allowed), ("B", True))
        detail = parse_native_action_detail(fenced, allowed)
        self.assertTrue(detail["normalization_used"])
        self.assertEqual(detail["serialization_form"],
                         "single_lowercase_json_fence")
        for raw in ('prefix {"action":"A"}', '{"action":"A","why":"x"}',
                    '{"action":"A","action":"B"}', '{"action":"C"}', "A", "",
                    '```json\n{"action":"A"}\n``` trailing',
                    '```JSON\n{"action":"A"}\n```'):
            self.assertEqual(parse_native_action(raw, allowed), (None, False))

    def test_transfer_decode_preserves_raw_bytes_before_parser(self):
        source = (Path(__file__).parents[1] / "src" /
                  "continuity_transfer_inference_worker.py").read_text()
        self.assertIn(
            "self.tokenizer.decode(suffix, skip_special_tokens=True),", source)
        self.assertNotIn(
            "self.tokenizer.decode(suffix, skip_special_tokens=True).strip()", source)

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

    def test_self_consistent_recovery_audit_tamper_rejects_receipt(self):
        manifest = json.loads(self.manifest.read_text())
        row = next(item for item in manifest["checkpoints"]
                   if item["identity_kind"] == "adapted")
        directory = self.checkpoint_root / row["relative_directory"]
        config = json.loads((directory / "adapter_config.json").read_text())
        audit_path = directory / "lineage" / "TRAINING_AUDIT.json"
        audit = json.loads(audit_path.read_text())
        endpoint = audit["three_residual_stage_chain"][-1]["recovery_audit"]
        endpoint["physical_tokens"] += 1
        payload = dict(endpoint); payload.pop("recovery_audit_sha256")
        endpoint["recovery_audit_sha256"] = digest(payload)
        summaries = [item["recovery_audit"]
                     for item in audit["three_residual_stage_chain"]]
        audit["cumulative_physical_tokens"] = sum(
            item["physical_tokens"] for item in summaries)
        audit["recovery_audit_chain_sha256"] = digest(summaries)
        payload = dict(audit); payload.pop("audit_payload_sha256")
        audit["audit_payload_sha256"] = digest(payload)
        dump(audit_path, audit)
        changed = dict(row)
        changed["training_audit_sha256"] = file_sha256(audit_path)
        with self.assertRaisesRegex(PermissionError, "recovery audit"):
            verify_training_lineage(changed, directory, config)

    def test_self_hashed_recovery_audit_rejects_bool_nan_and_infinity(self):
        manifest = json.loads(self.manifest.read_text())
        row = next(item for item in manifest["checkpoints"]
                   if item["identity_kind"] == "adapted")
        directory = self.checkpoint_root / row["relative_directory"]
        config = json.loads((directory / "adapter_config.json").read_text())
        audit_path = directory / "lineage" / "TRAINING_AUDIT.json"
        original = json.loads(audit_path.read_text())
        for label, value in (("bool", True), ("nan", float("nan")),
                             ("positive_inf", float("inf")),
                             ("negative_inf", float("-inf"))):
            audit = copy.deepcopy(original)
            endpoint = audit["three_residual_stage_chain"][-1][
                "recovery_audit"]
            endpoint["minimum_positive_gradient_norm"] = value
            payload = dict(endpoint); payload.pop("recovery_audit_sha256")
            endpoint["recovery_audit_sha256"] = digest(payload)
            summaries = [item["recovery_audit"]
                         for item in audit["three_residual_stage_chain"]]
            audit["minimum_positive_gradient_norm"] = value
            audit["recovery_audit_chain_sha256"] = digest(summaries)
            payload = dict(audit); payload.pop("audit_payload_sha256")
            audit["audit_payload_sha256"] = digest(payload)
            dump(audit_path, audit)
            changed = dict(row)
            changed["training_audit_sha256"] = file_sha256(audit_path)
            with self.subTest(label=label), self.assertRaises(PermissionError):
                verify_training_lineage(changed, directory, config)
        for field, value in (
                ("grad_scaler_initial_scale", True),
                ("minimum_grad_scaler_scale", True),
                ("max_overflow_adjustments", False)):
            audit = copy.deepcopy(original)
            endpoint = audit["three_residual_stage_chain"][-1][
                "recovery_audit"]
            endpoint[field] = value
            payload = dict(endpoint); payload.pop("recovery_audit_sha256")
            endpoint["recovery_audit_sha256"] = digest(payload)
            summaries = [item["recovery_audit"]
                         for item in audit["three_residual_stage_chain"]]
            audit["recovery_audit_chain_sha256"] = digest(summaries)
            payload = dict(audit); payload.pop("audit_payload_sha256")
            audit["audit_payload_sha256"] = digest(payload)
            dump(audit_path, audit)
            changed = dict(row)
            changed["training_audit_sha256"] = file_sha256(audit_path)
            with self.subTest(field=field), self.assertRaises(PermissionError):
                verify_training_lineage(changed, directory, config)
        dump(audit_path, original)

    def test_recovery_scale_chain_rejects_self_consistent_tamper(self):
        wrong_initial = recovery_receipt_fields(update_count=2)
        for update in wrong_initial["updates"]:
            attempt = update["grad_scaler_attempts"][0]
            attempt["scale_before"] = 2.0
            attempt["scale_after"] = 2.0

        growth = recovery_receipt_fields(update_count=2)
        growth["updates"][0]["grad_scaler_attempts"][0]["scale_after"] = 2.0

        zero_norm = recovery_receipt_fields(update_count=2)
        zero_norm["updates"][0]["gradient_norm_before_clip"] = 0.0
        zero_norm["updates"][0]["grad_scaler_attempts"][0][
            "gradient_norm_before_clip"] = 0.0
        zero_norm["updates"][0]["gradient_norm_positive"] = False

        zero_density = recovery_receipt_fields(update_count=2)
        zero_density["updates"][0][
            "active_branch_finite_nonzero_gradient_density"] = 0.0

        for label, receipt in (
                ("initial", wrong_initial),
                ("growth", growth),
                ("zero_norm", zero_norm),
                ("zero_density", zero_density)):
            with self.subTest(label=label), self.assertRaises(PermissionError):
                _training_recovery_summary(receipt)

    def test_fixed_scale_recovery_rejects_bool_nan_and_infinity(self):
        logical_bool = recovery_receipt_fields(update_count=1)
        update = logical_bool["updates"][0]
        update["logical_tokens"] = True
        update["initial_physical_tokens"] = True
        update["physical_tokens_processed"] = True
        logical_bool["tokens_seen"] = True
        logical_bool["physical_tokens_processed"] = True

        count_bool = recovery_receipt_fields(update_count=1)
        update = count_bool["updates"][0]
        attempt = update["grad_scaler_attempts"][0]
        update["active_branch_finite_nonzero_gradient_element_count"] = True
        attempt["diagnostic"][
            "active_branch_finite_nonzero_gradient_element_count"] = True
        attempt["diagnostic"]["trainable_gradient_element_count"] = 2
        update["active_branch_finite_nonzero_gradient_density"] = 0.5
        attempt["diagnostic"][
            "active_branch_finite_nonzero_gradient_density"] = 0.5

        variants = [("logical_bool", logical_bool),
                    ("count_bool", count_bool)]
        static_initial_bool = recovery_receipt_fields(update_count=1)
        static_initial_bool["grad_scaler_recovery"]["initial_scale"] = True
        variants.append(("static_initial_scale_bool", static_initial_bool))
        static_minimum_bool = recovery_receipt_fields(update_count=1)
        static_minimum_bool["grad_scaler_recovery"]["minimum_scale"] = True
        variants.append(("static_minimum_scale_bool", static_minimum_bool))
        static_maximum_bool = recovery_receipt_fields(update_count=1)
        static_maximum_bool["grad_scaler_recovery"][
            "max_overflow_adjustments_per_partition"] = False
        variants.append(("static_max_adjustments_bool", static_maximum_bool))
        attempt_scale_bool = recovery_receipt_fields(update_count=1)
        attempt_scale_bool["updates"][0]["grad_scaler_attempts"][0][
            "scale_before"] = True
        attempt_scale_bool["updates"][0]["grad_scaler_attempts"][0][
            "scale_after"] = True
        variants.append(("attempt_scale_bool", attempt_scale_bool))
        for label, value in (("bool", True), ("nan", float("nan")),
                             ("positive_inf", float("inf")),
                             ("negative_inf", float("-inf"))):
            norm = recovery_receipt_fields(update_count=1)
            norm["updates"][0]["gradient_norm_before_clip"] = value
            norm["updates"][0]["grad_scaler_attempts"][0][
                "gradient_norm_before_clip"] = value
            variants.append((f"norm_{label}", norm))
            density = recovery_receipt_fields(update_count=1)
            density["updates"][0][
                "active_branch_finite_nonzero_gradient_density"] = value
            density["updates"][0]["grad_scaler_attempts"][0][
                "diagnostic"][
                    "active_branch_finite_nonzero_gradient_density"] = value
            variants.append((f"density_{label}", density))
            attempt_scale = recovery_receipt_fields(update_count=1)
            attempt_scale["updates"][0]["grad_scaler_attempts"][0][
                "scale_before"] = value
            attempt_scale["updates"][0]["grad_scaler_attempts"][0][
                "scale_after"] = value
            variants.append((f"attempt_scale_{label}", attempt_scale))
        for label, receipt in variants:
            with self.subTest(label=label), self.assertRaises(PermissionError):
                _training_recovery_summary(receipt)

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
            "serialization_contract_version": "research4-action-serialization-v2",
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
