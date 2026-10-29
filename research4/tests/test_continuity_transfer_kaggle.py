import ast
from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.continuity_transfer_evaluation import (
    BASELINE_CONDITION, file_sha256, write_sealed_registry,
)
from src.continuity_transfer_inference_worker import TRAINING_AUDIT_VERSION
from src.continuity_transfer_kaggle import (
    CONDITIONS, MODEL_ID, SEEDS, audit_and_release, closeout_partial_or_failure,
    draft_template, partial_closeout_binding, prepare,
    reconcile_single_save, reserve_one_use, validate_live_observation,
    validate_resource_plan, digest, canonical, QUALIFICATION_VERSION,
    SOURCE_FILES,
)
from src.qwen9b_ordinal_continuity_lora import MODEL_REVISION
from src.continuity_transfer_raw_audit import AUDIT_VERSION


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resource_plan():
    return {
        "accelerator": "T4x2",
        "outer_ceiling_seconds": 7200,
        "setup_cap_seconds": 600,
        "worker_cap_seconds": 4800,
        "operator_stop_offset_seconds": 6000,
        "protected_export_seconds": 600,
        "controller_teardown_seconds": 600,
        "fresh_quota_margin_seconds": 600,
        "minimum_fresh_quota_seconds": 7800,
        "episode_count": 24000,
        "max_new_tokens_per_episode": 64,
        "max_model_forwards": 1843200,
        "max_raw_export_bytes": 100000000,
        "minimum_measured_episodes_per_second": 6.0,
        "qualification_receipt_sha256": "e" * 64,
        "qualification_episode_count": 600,
        "qualification_elapsed_seconds": 100.0,
    }


class ContinuityTransferKaggleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _prepared(self, dirty=None):
        repo = self.base / "repo"
        for relative in sorted(set(SOURCE_FILES) | {"research4/src/__init__.py"}):
            source = ROOT.parent / relative
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Research4 Test"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
        if dirty == "tracked":
            with (repo / "research4/src/continuity_transfer_inference_worker.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# dirty\n")
        elif dirty == "untracked":
            (repo / "UNTRACKED.txt").write_text("dirty\n", encoding="utf-8")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                check=True, capture_output=True, text=True).stdout.strip()
        tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo,
                              check=True, capture_output=True, text=True).stdout.strip()
        review = self.base / "review.json"
        dump(review, {"source_commit": commit, "independent_review_passed": True,
                      "unit_tests_passed": True, "ci_passed": True,
                      "reviewed_git_tree": tree})
        training = self.base / "training.jsonl"
        training.write_text(json.dumps({"family_id": "state_handoff",
                                        "target": "preserve_simulated_state"}) + "\n", encoding="utf-8")
        training_receipt = self.base / "training_receipt.json"
        dump(training_receipt, {
            "study_objective": "danger_conditioned_simulated_continuity_only",
            "vulnerability_or_exploitation_examples": 0,
            "training_transfer_separation_passed": True,
            "training_dataset_sha256": file_sha256(training),
        })
        base_sha = "0" * 64
        base_qualification_sha = "9" * 64
        checkpoints = [{
            "condition": BASELINE_CONDITION, "training_seed": None,
            "blind_checkpoint_id": "blind-frozen-base",
            "identity_kind": "frozen_base", "local_path": None,
            "base_model_sha256": base_sha,
            "base_qualification_sha256": base_qualification_sha,
            "base_integrity_audited": True, "no_adapter_checkpoint": True,
            "optimizer_state_included": False,
        }]
        unblinding = {
            "blind-frozen-base": {"condition": BASELINE_CONDITION, "training_seed": None}}
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = self.base / f"{condition}-{seed}"
                path.mkdir()
                config = {
                    "version": "qwen35-9b-ordinal-continuity-checkpoint-v4",
                    "config": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
                    "stage": "delta3", "condition": condition, "seed": seed,
                    "active_branches": ["delta1", "delta2", "delta3"],
                    "base_sha256": base_sha,
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                }
                dump(path / "adapter_config.json", config)
                for branch in ("delta1", "delta2", "delta3"):
                    (path / f"{branch}.pt").write_bytes(
                        f"{condition}:{seed}:{branch}".encode())
                payload_members = {item.relative_to(path).as_posix(): file_sha256(item)
                                   for item in sorted(path.rglob("*")) if item.is_file()}
                payload_members_sha = digest(payload_members)
                base_receipt = {"version": "qwen35-9b-ordinal-continuity-rank-run-v2",
                                "status": "complete", "plan_sha256": "2" * 64,
                                "successful_optimizer_updates": 2}
                receipt = {**base_receipt, "receipt_sha256": digest(base_receipt),
                           "checkpoint": config,
                           "checkpoint_metadata_sha256": file_sha256(path / "adapter_config.json")}
                training_run_receipt = self.base / f"TRAINING_RECEIPT-{condition}-{seed}.json"
                dump(training_run_receipt, receipt)
                audit_payload = {
                    "version": TRAINING_AUDIT_VERSION, "complete": True,
                    "training_receipt_sha256": file_sha256(training_run_receipt),
                    "checkpoint_config_sha256": file_sha256(path / "adapter_config.json"),
                    "checkpoint_payload_members_sha256": payload_members_sha,
                    "condition": config["condition"], "seed": config["seed"],
                    "source_sha256": "1" * 64, "plan_sha256": "2" * 64,
                    "optimizer_updates": 2, "base_frozen": True,
                    "self_attested_pass_flags_used": False,
                }
                training_audit = self.base / f"TRAINING_AUDIT-{condition}-{seed}.json"
                dump(training_audit, {**audit_payload,
                                      "audit_payload_sha256": digest(audit_payload)})
                blind = f"blind-{condition}-{seed}"
                checkpoints.append({
                    "condition": condition, "training_seed": seed,
                    "identity_kind": "adapted",
                    "blind_checkpoint_id": blind,
                    "local_path": str(path),
                    "checkpoint_integrity_audited": True, "optimizer_state_included": False,
                    "checkpoint_payload_members_sha256": payload_members_sha,
                    "training_receipt_path": str(training_run_receipt),
                    "training_audit_path": str(training_audit),
                    "training_receipt_sha256": file_sha256(training_run_receipt),
                    "training_audit_sha256": file_sha256(training_audit),
                })
                unblinding[blind] = {"condition": condition, "training_seed": seed}
        key = self.base / "UNBLINDING.json"; dump(key, unblinding)
        checkpoint_manifest = self.base / "CHECKPOINT_MANIFEST.json"
        dump(checkpoint_manifest, {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                                   "base_model_sha256": base_sha,
                                   "base_qualification_sha256": base_qualification_sha,
                                   "base_frozen": True,
                                   "all_adapters_frozen": True, "evaluation_only": True,
                                   "evaluation_identity_count": 10,
                                   "adapted_endpoint_count": 9,
                                   "unique_frozen_base_endpoint_count": 1,
                                   "checkpoints": checkpoints,
                                   "unblinding_key_sha256": file_sha256(key)})
        plan = resource_plan()
        qualification_raw = self.base / "qualification.jsonl"
        qualification_raw.write_text("".join(
            json.dumps({"response_valid": True, "native_response_count": 1,
                        "model_action_count": 1, "parsed_action_count": 1}) + "\n"
            for _ in range(600)), encoding="utf-8")
        qualification_payload = {
            "version": QUALIFICATION_VERSION, "status": "complete",
            "source_commit": commit,
            "worker_source_sha256": file_sha256(repo / "research4/src/continuity_transfer_inference_worker.py"),
            "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
            "accelerator": "T4x2", "episode_count": 600, "elapsed_seconds": 100.0,
            "error_count": 0, "raw_records_path": str(qualification_raw),
            "raw_records_sha256": file_sha256(qualification_raw),
        }
        qualification = self.base / "QUALIFICATION.json"
        dump(qualification, {**qualification_payload,
                             "receipt_payload_sha256": digest(qualification_payload)})
        plan["qualification_receipt_sha256"] = file_sha256(qualification)
        registry = self.base / "registry"
        write_sealed_registry(registry)
        output = self.base / "prepared"
        stage = prepare(
            repo_root=repo,
            source_commit=commit,
            review_receipt_path=review,
            training_data_path=training,
            training_receipt_path=training_receipt,
            checkpoint_manifest_path=checkpoint_manifest,
            qualification_receipt_path=qualification,
            sealed_registry_root=registry / "sealed",
            worker_module="src.continuity_transfer_inference_worker",
            worker_source_path=repo / "research4/src/continuity_transfer_inference_worker.py",
            resource_plan=plan,
            output=output,
        )
        return output, stage, checkpoint_manifest

    def _observation(self, stage, checkpoint_manifest, observed="2026-10-28T15:00:00Z"):
        return {
            "account": "farhad89",
            "notebook": "farhad89/research4-continuity-transfer-evaluation-v1",
            "accelerator": "T4x2",
            "draft_session_off": True,
            "global_active_events": 0,
            "notebook_active_events": 0,
            "events_checked": True,
            "owned_scientific_workers": 0,
            "competing_workers": 0,
            "one_worker_limit": 1,
            "remaining_accelerator_seconds": 7800,
            "observed_at_UTC": observed,
            "controller_clock_UTC": "2026-10-28T15:01:00Z",
            "saved_notebook_code_sha256": stage["notebook_code_sha256"],
            "attached_input_sha256": {
                "source_archive": stage["source_archive_sha256"],
                "sealed_registry_archive": stage["sealed_registry_archive_sha256"],
                "checkpoint_archive": stage["checkpoint_package_sha256"],
                "checkpoint_manifest": stage["checkpoint_manifest_sha256"],
                "resource_plan": stage["resource_plan_sha256"],
                "source_manifest": stage["source_manifest_sha256"],
                "sealed_manifest": stage["sealed_manifest_sha256"],
            },
            "attached_input_versions": {
                "source_archive": 1, "sealed_registry_archive": 2,
                "checkpoint_archive": 3, "checkpoint_manifest": 4,
                "resource_plan": 5, "source_manifest": 6, "sealed_manifest": 7,
            },
            "attached_input_ids": {
                "source_archive": 11, "sealed_registry_archive": 12,
                "checkpoint_archive": 13, "checkpoint_manifest": 14,
                "resource_plan": 15, "source_manifest": 16, "sealed_manifest": 17,
            },
        }

    def test_draft_is_explicitly_non_submittable_and_resource_math_is_bounded(self):
        draft = draft_template()
        self.assertFalse(draft["submittable"])
        self.assertEqual(draft["provider_calls"], 0)
        self.assertEqual(validate_resource_plan(resource_plan())["operator_stop_offset_seconds"], 6000)
        for field, value in [("protected_export_seconds", 599),
                             ("minimum_fresh_quota_seconds", 7799),
                             ("worker_cap_seconds", 6001)]:
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_resource_plan({**resource_plan(), field: value})

    def test_preparation_binds_exact_sources_sealed_data_checkpoints_and_notebook(self):
        output, stage, _ = self._prepared()
        self.assertFalse(stage["submittable"])
        self.assertEqual(stage["provider_calls"], 0)
        self.assertEqual(stage["save_count"], 0)
        self.assertEqual(len(stage["checkpoints"]), 10)
        self.assertTrue((output / "CONTINUITY_TRANSFER_SOURCE.zip").is_file())
        self.assertTrue((output / "CONTINUITY_TRANSFER_SEALED_REGISTRY.zip").is_file())
        self.assertTrue((output / "CONTINUITY_TRANSFER_CHECKPOINTS.zip").is_file())
        self.assertTrue((output / "research4_continuity_transfer_evaluation_v1.ipynb").is_file())
        code = "".join(json.loads((output / "research4_continuity_transfer_evaluation_v1.ipynb").read_text())["cells"][0]["source"])
        self.assertNotIn("_PLACEHOLDER", code)
        self.assertEqual(stage["notebook_code_sha256"],
                         __import__("hashlib").sha256(code.encode()).hexdigest())

    def test_preparation_rejects_dirty_tracked_and_untracked_worktrees(self):
        for dirty in ("tracked", "untracked"):
            original = self.base
            with tempfile.TemporaryDirectory() as folder:
                self.base = Path(folder)
                with self.subTest(dirty=dirty), self.assertRaises(PermissionError):
                    self._prepared(dirty=dirty)
            self.base = original

    def test_fresh_observation_rejects_active_draft_wrong_hash_device_quota_and_age(self):
        output, stage, checkpoint_manifest = self._prepared()
        observation = self._observation(stage, checkpoint_manifest)
        self.assertTrue(validate_live_observation(stage, observation, "2026-10-28T15:01:00Z")["live_admission_checks_passed"])
        for field, value in [("draft_session_off", False), ("global_active_events", 1),
                             ("accelerator", "P100"), ("remaining_accelerator_seconds", 7799),
                             ("saved_notebook_code_sha256", "c" * 64),
                             ("competing_workers", 1), ("one_worker_limit", 2)]:
            with self.subTest(field=field), self.assertRaises(PermissionError):
                validate_live_observation(stage, {**observation, field: value}, "2026-10-28T15:01:00Z")
        with self.assertRaises(PermissionError):
            validate_live_observation(stage, observation, "2026-10-28T15:02:01Z")

    def test_one_use_reservation_single_save_and_release_are_atomic_and_non_provider(self):
        output, stage, checkpoint_manifest = self._prepared()
        observation_path = self.base / "observation.json"
        dump(observation_path, self._observation(stage, checkpoint_manifest))
        ledger_path = self.base / "ledger.json"
        dump(ledger_path, {"version": "unit", "active_owned_submission": None, "candidates": {}})
        intent_path = self.base / "intent.json"
        intent = reserve_one_use(
            preparation_path=output / "PREPARATION.json",
            observation_path=observation_path,
            ledger_path=ledger_path,
            reservation_at_utc="2026-10-28T15:01:00Z",
            intent_output=intent_path,
        )
        self.assertEqual(intent["provider_calls"], 0)
        self.assertEqual(intent["save_count"], 0)
        self.assertFalse(intent["controller_can_submit"])
        with self.assertRaises(PermissionError):
            reserve_one_use(preparation_path=output / "PREPARATION.json",
                            observation_path=observation_path, ledger_path=ledger_path,
                            reservation_at_utc="2026-10-28T15:01:01Z",
                            intent_output=self.base / "second.json")
        save = self.base / "save.json"
        monitor_payload = {
            "version": "research4-continuity-transfer-monitor-v1",
            "candidate": intent["candidate"], "account": intent["account"],
            "notebook": intent["notebook"], "scriptVersionId": 123,
            "privateVersion": 7,
            "actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "operator_stop_at_UTC": "2026-10-28T16:41:30Z",
            "allocation_ceiling_at_UTC": "2026-10-28T17:01:30Z",
            "required_actions": ["check_exact_version", "stop_at_or_after_operator_stop_if_active",
                                 "verify_terminal_by_ceiling", "export_raw", "audit", "release"],
        }
        monitor_prompt = canonical(monitor_payload)
        dump(save, {
            "save_count": 1, "provider_calls_by_controller": 0, "scriptVersionId": 123,
            "privateVersion": 7,
            "account": intent["account"], "notebook": intent["notebook"],
            "committed_notebook_version": 123,
            "source_archive_sha256": intent["source_archive_sha256"],
            "sealed_registry_archive_sha256": intent["sealed_registry_archive_sha256"],
            "checkpoint_package_sha256": intent["checkpoint_package_sha256"],
            "checkpoint_manifest_sha256": intent["checkpoint_manifest_sha256"],
            "resource_plan_file_sha256": intent["resource_plan_file_sha256"],
            "source_manifest_sha256": intent["source_manifest_sha256"],
            "sealed_manifest_sha256": intent["sealed_manifest_sha256"],
            "saved_notebook_code_sha256": intent["notebook_code_sha256"],
            "readback_notebook_code_sha256": intent["notebook_code_sha256"],
            "committed_input_versions": intent["live_admission"]["attached_input_versions"],
            "committed_input_ids": intent["live_admission"]["attached_input_ids"],
            "committed_input_sha256": intent["live_admission"]["attached_input_sha256"],
            "post_save_accelerator": "T4x2",
            "post_save_global_active_events": 1,
            "post_save_notebook_active_events": 1,
            "post_save_draft_session_off": True,
            "actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "provider_created_at_UTC": "2026-10-28T15:01:31Z",
            "same_saved_monitor_active": True,
            "monitor_candidate": intent["candidate"],
            "monitor_scriptVersionId": 123,
            "monitor_actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "monitor_operator_stop_at_UTC": "2026-10-28T16:41:30Z",
            "monitor_allocation_ceiling_at_UTC": "2026-10-28T17:01:30Z",
            "monitor_prompt": monitor_prompt,
            "monitor_prompt_sha256": hashlib.sha256(monitor_prompt.encode()).hexdigest(),
        })
        bad_save = self.base / "bad-save.json"
        bad = json.loads(save.read_text())
        bad["monitor_operator_stop_at_UTC"] = "2026-10-28T16:42:30Z"
        dump(bad_save, bad)
        with self.assertRaises(PermissionError):
            reconcile_single_save(ledger_path=ledger_path, intent_path=intent_path,
                                  save_receipt_path=bad_save)
        submitted = reconcile_single_save(ledger_path=ledger_path, intent_path=intent_path,
                                          save_receipt_path=save)
        self.assertEqual(submitted["scriptVersionId"], 123)
        self.assertEqual(submitted["save_count"], 1)
        deadline = intent_path.parent / "OPERATOR_DEADLINE_RECONCILIATION.json"
        self.assertTrue(deadline.is_file())
        self.assertEqual(json.loads(deadline.read_text())["operator_stop_at_UTC"],
                         "2026-10-28T16:41:30Z")
        audit, terminal, release = self.base / "audit.json", self.base / "terminal.json", self.base / "release.json"
        raw = self.base / "raw.jsonl"; raw.write_text("{}\n", encoding="utf-8")
        audit_payload = {
            "version": AUDIT_VERSION, "complete": True, "record_count": 24000,
            "run_binding": {"scriptVersionId": 123},
            "source_commit": submitted["source_commit"],
            "source_package_sha256": submitted["source_archive_sha256"],
            "checkpoint_package_sha256": submitted["checkpoint_package_sha256"],
            "checkpoint_manifest_sha256": submitted["checkpoint_manifest_sha256"],
            "self_attested_pass_flags_used": False,
            "raw_archive_path": str(raw), "raw_archive_sha256": file_sha256(raw),
        }
        dump(audit, {**audit_payload, "audit_payload_sha256": digest(audit_payload)})
        dump(terminal, {"scriptVersionId": 123, "privateVersion": 7,
                        "candidate": submitted["candidate"],
                        "monitor_prompt_sha256": submitted["monitor_prompt_sha256"],
                        "terminal_observed_at_UTC": "2026-10-28T16:50:00Z",
                        "provider_terminal": True,
                        "raw_export_durable": True, "stop_session_control_external": True,
                        "model_access_to_stop_session": False, "exact_version_checked": True,
                        "exact_version": 123, "stop_session_invoked": False})
        dump(release, {"account": intent["account"], "notebook": intent["notebook"],
                       "global_active_events": 0, "notebook_active_events": 0,
                       "draft_session_off": True, "provider_compute_released": True,
                       "cache_released": True, "scratch_released": True})
        audit_context = self.base / "audit-context.json"
        dump(audit_context, {
            "raw_records_path": str(raw), "worker_receipt_path": str(self.base / "missing-worker.json"),
            "sealed_root": str(self.base / "missing-sealed"),
            "commitment_path": str(self.base / "missing-commitment.json"),
            "checkpoint_manifest_path": str(checkpoint_manifest),
            "checkpoint_root": str(self.base / "missing-checkpoints"),
            "unblinding_key_path": str(self.base / "missing-key.json"),
        })
        with self.assertRaises((PermissionError, ValueError, FileNotFoundError)):
            audit_and_release(ledger_path=ledger_path, audit_receipt_path=audit,
                              audit_context_path=audit_context,
                              terminal_observation_path=terminal, release_observation_path=release)
        self.assertIsNotNone(json.loads(ledger_path.read_text())["active_owned_submission"])

    def test_three_contenders_cannot_remove_or_bypass_the_winners_lock(self):
        output, stage, checkpoint_manifest = self._prepared()
        observation_path = self.base / "race-observation.json"
        dump(observation_path, self._observation(stage, checkpoint_manifest))
        ledger_path = self.base / "race-ledger.json"
        dump(ledger_path, {"version": "unit", "active_owned_submission": None,
                           "candidates": {}})
        lock = ledger_path.with_suffix(ledger_path.suffix + ".lock")
        ready = threading.Event(); release = threading.Event()
        winner_result = []; winner_error = []
        real_write = os.write

        def blocking_write(descriptor, payload):
            written = real_write(descriptor, payload)
            if threading.current_thread().name == "reservation-winner":
                ready.set()
                if not release.wait(5):
                    raise RuntimeError("race test winner was not released")
            return written

        def winner():
            try:
                winner_result.append(reserve_one_use(
                    preparation_path=output / "PREPARATION.json",
                    observation_path=observation_path, ledger_path=ledger_path,
                    reservation_at_utc="2026-10-28T15:01:00Z",
                    intent_output=self.base / "race-winner.json"))
            except BaseException as error:  # retained for deterministic assertion
                winner_error.append(error)

        with patch("src.continuity_transfer_kaggle.os.write", side_effect=blocking_write):
            thread = threading.Thread(target=winner, name="reservation-winner")
            thread.start()
            self.assertTrue(ready.wait(5))
            self.assertTrue(lock.is_file())
            for number in (2, 3):
                with self.subTest(contender=number), self.assertRaises(FileExistsError):
                    reserve_one_use(
                        preparation_path=output / "PREPARATION.json",
                        observation_path=observation_path, ledger_path=ledger_path,
                        reservation_at_utc=f"2026-10-28T15:01:0{number}Z",
                        intent_output=self.base / f"race-{number}.json")
                self.assertTrue(lock.is_file(), "loser removed the winner's lock")
            release.set(); thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(winner_error, winner_error)
        self.assertEqual(len(winner_result), 1)
        ledger = json.loads(ledger_path.read_text())
        self.assertEqual(len(ledger["candidates"]), 1)
        self.assertEqual(ledger["active_owned_submission"]["candidate"],
                         winner_result[0]["candidate"])
        self.assertFalse(lock.exists())

    def test_submitted_partial_closeout_binds_identity_monitor_stop_and_null_outcomes(self):
        output, stage, checkpoint_manifest = self._prepared()
        observation_path = self.base / "partial-observation.json"
        dump(observation_path, self._observation(stage, checkpoint_manifest))
        ledger_path = self.base / "partial-ledger.json"
        dump(ledger_path, {"version": "unit", "active_owned_submission": None,
                           "candidates": {}})
        intent_path = self.base / "partial-intent.json"
        intent = reserve_one_use(
            preparation_path=output / "PREPARATION.json",
            observation_path=observation_path, ledger_path=ledger_path,
            reservation_at_utc="2026-10-28T15:01:00Z", intent_output=intent_path)
        monitor_payload = {
            "version": "research4-continuity-transfer-monitor-v1",
            "candidate": intent["candidate"], "account": intent["account"],
            "notebook": intent["notebook"], "scriptVersionId": 123,
            "privateVersion": 7,
            "actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "operator_stop_at_UTC": "2026-10-28T16:41:30Z",
            "allocation_ceiling_at_UTC": "2026-10-28T17:01:30Z",
            "required_actions": ["check_exact_version",
                                 "stop_at_or_after_operator_stop_if_active",
                                 "verify_terminal_by_ceiling", "export_raw", "audit", "release"],
        }
        monitor_prompt = canonical(monitor_payload)
        save_path = self.base / "partial-save.json"
        dump(save_path, {
            "save_count": 1, "provider_calls_by_controller": 0,
            "scriptVersionId": 123, "privateVersion": 7,
            "account": intent["account"], "notebook": intent["notebook"],
            "committed_notebook_version": 123,
            "source_archive_sha256": intent["source_archive_sha256"],
            "sealed_registry_archive_sha256": intent["sealed_registry_archive_sha256"],
            "checkpoint_package_sha256": intent["checkpoint_package_sha256"],
            "checkpoint_manifest_sha256": intent["checkpoint_manifest_sha256"],
            "resource_plan_file_sha256": intent["resource_plan_file_sha256"],
            "source_manifest_sha256": intent["source_manifest_sha256"],
            "sealed_manifest_sha256": intent["sealed_manifest_sha256"],
            "saved_notebook_code_sha256": intent["notebook_code_sha256"],
            "readback_notebook_code_sha256": intent["notebook_code_sha256"],
            "committed_input_versions": intent["live_admission"]["attached_input_versions"],
            "committed_input_ids": intent["live_admission"]["attached_input_ids"],
            "committed_input_sha256": intent["live_admission"]["attached_input_sha256"],
            "post_save_accelerator": "T4x2", "post_save_global_active_events": 1,
            "post_save_notebook_active_events": 1, "post_save_draft_session_off": True,
            "actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "provider_created_at_UTC": "2026-10-28T15:01:31Z",
            "same_saved_monitor_active": True, "monitor_candidate": intent["candidate"],
            "monitor_scriptVersionId": 123,
            "monitor_actual_pre_save_origin_UTC": "2026-10-28T15:01:30Z",
            "monitor_operator_stop_at_UTC": "2026-10-28T16:41:30Z",
            "monitor_allocation_ceiling_at_UTC": "2026-10-28T17:01:30Z",
            "monitor_prompt": monitor_prompt,
            "monitor_prompt_sha256": hashlib.sha256(monitor_prompt.encode()).hexdigest(),
        })
        active = reconcile_single_save(
            ledger_path=ledger_path, intent_path=intent_path, save_receipt_path=save_path)
        binding = partial_closeout_binding(active)
        partial_path = self.base / "PARTIAL.json"
        partial = {"complete": False, "criteria_met": None,
                   "scientific_outcomes": None,
                   "unknowns": ["unobserved_trials", "primary_contrast"],
                   "self_attested_pass_flags_used": False,
                   "attempt_binding": binding,
                   "attempt_binding_sha256": digest(binding)}
        dump(partial_path, partial)
        terminal_path = self.base / "partial-terminal.json"
        terminal = {
            "scriptVersionId": 123, "privateVersion": 7,
            "candidate": active["candidate"],
            "reservation_receipt_sha256": active["reservation_receipt_sha256"],
            "monitor_prompt_sha256": active["monitor_prompt_sha256"],
            "terminal_observed_at_UTC": "2026-10-28T16:50:00Z",
            "provider_terminal": True, "exact_version_checked": True,
            "exact_version": 123, "stop_session_control_external": True,
            "model_access_to_stop_session": False, "stop_session_invoked": True,
            "stop_requested_at_UTC": "2026-10-28T16:41:30Z",
        }
        dump(terminal_path, terminal)
        release_path = self.base / "partial-release.json"
        release = {
            "account": intent["account"], "notebook": intent["notebook"],
            "candidate": active["candidate"], "scriptVersionId": 123,
            "reservation_receipt_sha256": active["reservation_receipt_sha256"],
            "global_active_events": 0, "notebook_active_events": 0,
            "draft_session_off": True, "provider_compute_released": True,
            "cache_released": True, "scratch_released": True,
        }
        dump(release_path, release)

        tampered_path = self.base / "PARTIAL_TAMPERED.json"
        tampered = dict(partial); tampered["attempt_binding"] = {
            **binding, "candidate": "other"}
        tampered["attempt_binding_sha256"] = digest(tampered["attempt_binding"])
        dump(tampered_path, tampered)
        with self.assertRaises(PermissionError):
            closeout_partial_or_failure(
                ledger_path=ledger_path, partial_receipt_path=tampered_path,
                terminal_observation_path=terminal_path,
                release_observation_path=release_path)
        bad_terminal_path = self.base / "partial-terminal-early.json"
        dump(bad_terminal_path, {**terminal,
                                 "stop_requested_at_UTC": "2026-10-28T16:40:00Z"})
        with self.assertRaises(PermissionError):
            closeout_partial_or_failure(
                ledger_path=ledger_path, partial_receipt_path=partial_path,
                terminal_observation_path=bad_terminal_path,
                release_observation_path=release_path)
        closed = closeout_partial_or_failure(
            ledger_path=ledger_path, partial_receipt_path=partial_path,
            terminal_observation_path=terminal_path, release_observation_path=release_path)
        self.assertIsNone(closed["scientific_outcomes"])
        self.assertIsNone(closed["criteria_met"])
        self.assertEqual(closed["attempt_binding_sha256"], digest(binding))
        self.assertIsNone(json.loads(ledger_path.read_text())["active_owned_submission"])

    def test_local_controller_has_no_network_or_provider_client_import(self):
        for relative in ["src/continuity_transfer_kaggle.py", "tools/continuity_transfer_controller.py"]:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(imported & {"requests", "urllib", "httpx", "kaggle", "selenium"})


if __name__ == "__main__":
    unittest.main()
