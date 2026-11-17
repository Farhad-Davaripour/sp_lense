"""Admission and release for an explicitly owned, receipt-bound serving handoff."""
from pathlib import Path
import json
import copy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import infrastructure_preservation_comparison as runner
from src import company_models as models
from tests.test_company_models import provider352_consumer_fixture, provider352_isolate_state


HEALTH = {"pid": 61991, "process_created_at": 1234.5, "pending_http_requests": 0}
RECEIPT = "e6d62efa8117a01952e4ba8794e03bcecd7009df33e5d3586d775fa2da1af5be"


def handoff():
    return {"status": "GAME_128K_ENDPOINT_HANDED_OFF", "owner_thread": runner.OWNER,
            "Game_requests_allowed": True, "handoff_sha256": RECEIPT,
            "service_pid": HEALTH["pid"], "service_process_created_at": HEALTH["process_created_at"]}


class PreservationRequestWindowTests(unittest.TestCase):
    def test_owned_handoff_and_available_window_admit_then_release_available(self):
        available = {**handoff(), "status": "AVAILABLE_NO_ACTIVE_COLLECTION", "owner_thread": None,
                     "Game_requests_allowed": False}
        for initial in (handoff(), available):
            with self.subTest(status=initial["status"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                window, lock = root / "WINDOW.json", root / "window.lock"
                runner.write_json(window, initial)
                with patch.object(runner, "WINDOW", window), patch.object(runner, "LOCK", lock), \
                        patch.object(runner, "current_local_binding", return_value={"readiness_receipt_sha256": RECEIPT}), \
                        patch.object(runner, "assert_local_binding", return_value=HEALTH), \
                        patch.object(runner, "idle_service", return_value=HEALTH):
                    with runner.request_window(root, "prospective-plan"):
                        active = runner.read_json(window)
                        self.assertEqual(active["status"], "ACTIVE_GAME_COLLECTION")
                        self.assertEqual(active["game_plan_sha256"], "prospective-plan")
                        self.assertTrue(lock.exists())
                released = runner.read_json(window)
                self.assertEqual(released["status"], "AVAILABLE_NO_ACTIVE_COLLECTION")
                self.assertIsNone(released["owner_thread"])
                self.assertFalse(released["Game_requests_allowed"])
                self.assertEqual(runner.read_json(root / "PREVIOUS_REQUEST_WINDOW.json"), initial)
                self.assertEqual(runner.read_json(root / "FINAL_IDLE_HEALTH.json"), HEALTH)
                self.assertFalse(lock.exists())

    def test_wrong_owner_receipt_or_permission_rejects_before_health(self):
        for changed in ({"owner_thread": "another-chat"}, {"handoff_sha256": "unreviewed"},
                        {"Game_requests_allowed": False}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                window, lock = root / "WINDOW.json", root / "window.lock"
                initial = {**handoff(), **changed}
                runner.write_json(window, initial)
                with patch.object(runner, "WINDOW", window), patch.object(runner, "LOCK", lock), \
                        patch.object(runner, "current_local_binding", return_value={"readiness_receipt_sha256": RECEIPT}), \
                        patch.object(runner, "assert_local_binding") as health:
                    with self.assertRaises(ValueError), runner.request_window(root, "prospective-plan"):
                        self.fail("A rejected handoff must not enter the execution window")
                health.assert_not_called()
                self.assertEqual(runner.read_json(window), initial)
                self.assertFalse(lock.exists())

    def test_owned_handoff_still_requires_exact_resident_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, lock = root / "WINDOW.json", root / "window.lock"
            runner.write_json(window, handoff())
            with patch.object(runner, "WINDOW", window), patch.object(runner, "LOCK", lock), \
                    patch.object(runner, "current_local_binding", return_value={"readiness_receipt_sha256": RECEIPT}), \
                    patch.object(runner, "assert_local_binding", return_value={**HEALTH, "pid": 1}):
                with self.assertRaisesRegex(ValueError, "different resident service"), runner.request_window(root, "plan"):
                    self.fail("A different process must not enter the execution window")
            self.assertEqual(runner.read_json(window), handoff())
            self.assertFalse(lock.exists())


class DormantProvider352WindowTests(unittest.TestCase):
    def test_v2_consumer_claim_never_rewrites_or_releases_roots_grant(self):
        fixture = provider352_consumer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grant, lock = root / "ROOT_GRANT.json", root / "claim.lock"
            original = fixture.records["current-window.json"]
            grant.write_bytes(original)
            with patch.object(runner, "WINDOW", grant), patch.object(runner, "LOCK", lock), \
                    patch.object(runner, "update_window", side_effect=AssertionError("Root grant is read-only")):
                with runner.provider352_request_window(root, fixture.pins["expected_game_plan_sha256"], fixture.consumer):
                    self.assertTrue(lock.exists())
                    self.assertEqual(grant.read_bytes(), original)
                self.assertFalse(lock.exists())
                self.assertEqual(grant.read_bytes(), original)
            closeout = json.loads((root / "PROVIDER352_CONSUMER_CLOSEOUT.json").read_bytes())
            self.assertIs(closeout["consumer_finished"], True)
            self.assertIs(closeout["root_window_release_confirmed"], False)
            self.assertIs(closeout["request_window_released"], False)

    def test_wrong_parent_plan_cannot_claim_a_window(self):
        fixture = provider352_consumer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "claim.lock"
            with patch.object(runner, "LOCK", lock), patch.object(runner, "write_json") as write:
                with self.assertRaisesRegex(ValueError, "Whole parent"):
                    with runner.provider352_request_window(root, "6" * 64, fixture.consumer):
                        self.fail("A different parent plan must not enter")
                self.assertFalse(lock.exists())
                self.assertEqual(write.call_args.args[0], root / "PROVIDER352_COLLECTION_STOP.json")
                fixture.health.assert_not_called()

    def test_child_receives_private_whole_parent_scope_without_command_or_prompt_fields(self):
        fixture = provider352_consumer_fixture()
        process = Mock()
        process.wait.return_value, process.poll.return_value = 0, 0
        batch = {"rules_version": "unit_fixture", "seed": 1, "models": ["qwen_base"]}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner.subprocess, "Popen", return_value=process) as start, \
                patch.object(runner, "audit_batch", return_value=([], {})):
            runner.run_batch(batch, Path(temporary) / "batch", Path("unused.env"), handoff_consumer=fixture.consumer)
        command = start.call_args.args[0]
        private = start.call_args.kwargs["env"][runner.PROVIDER352_CONTEXT_ENV]
        self.assertEqual(json.loads(private)["pins"]["expected_game_plan_sha256"], fixture.pins["expected_game_plan_sha256"])
        self.assertEqual(json.loads(private)["requester_chat"], fixture.consumer.requester_chat)
        self.assertFalse(any(private in arg for arg in command))
        self.assertEqual(set(batch), {"rules_version", "seed", "models"})

    def test_collection_preflight_precedes_hosted_dispatch_and_closeout_does_not_claim_release(self):
        # Minimal mock plan only; no study scenario, case, prompt, or seed data.
        candidate = {"models": ["qwen_base", "terra", "deepseek"], "episodes": 2,
                     "batches": [{"models": ["terra", "deepseek"], "seed": 1}, {"models": ["qwen_base"], "seed": 1}],
                     "seeds": [], "protocol": "unit_fixture", "horizon_hours": 48}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.write_json(root / "PLAN.json", candidate)
            fixture = provider352_consumer_fixture(runner.sha(root / "PLAN.json"))
            original = fixture.records["current-window.json"]
            calls = []
            def fake_batch(batch, directory, credentials, *, handoff_consumer):
                self.assertGreater(fixture.health.call_count, 0)
                self.assertIs(handoff_consumer, fixture.consumer)
                calls.append(batch["models"])
                return [{"evidence_status": "audited_complete"}], {"batch": directory.name, "exit_code": 0}
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "summarize", return_value={}), \
                    patch.object(runner, "print", create=True), \
                    patch.object(runner, "LOCK", root / "claim.lock"), \
                    patch.object(runner, "request_window", side_effect=AssertionError("No legacy window writes")), \
                    patch.object(runner, "update_window", side_effect=AssertionError("Root grant is read-only")), \
                    patch.object(runner, "idle_service", side_effect=AssertionError("No legacy health path")), \
                    patch.object(runner, "run_batch", side_effect=fake_batch):
                runner.execute_locked(root, Path("unused.env"), Path("unused.yaml"), "all", handoff_consumer=fixture.consumer)
            self.assertEqual(calls, [["terra", "deepseek"], ["qwen_base"]])
            self.assertEqual(fixture.records["current-window.json"], original)
            for name in ("PHASE_CLOSEOUT_all.json", "CLOSEOUT.json"):
                closeout = json.loads((root / name).read_bytes())
                self.assertIs(closeout["request_window_released"], False)
                self.assertIs(closeout["consumer_finished"], True)
                self.assertIs(closeout["root_window_release_confirmed"], False)

    def test_incomplete_child_stops_parent_and_a_fresh_consumer_cannot_reuse_that_grant(self):
        candidate = {"models": ["qwen_base", "terra", "deepseek"], "episodes": 2,
                     "batches": [{"models": ["terra", "deepseek"], "seed": 1}, {"models": ["qwen_base"], "seed": 1}],
                     "seeds": [], "protocol": "unit_fixture", "horizon_hours": 48}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.write_json(root / "PLAN.json", candidate)
            parent_sha = runner.sha(root / "PLAN.json")
            fixture = provider352_consumer_fixture(parent_sha)
            audited_cells = [{"evidence_status": "incomplete", "prefix_replay_passed": True, "survived_horizon": None}]
            audited_receipt = {"batch": "batch_01", "exit_code": 0, "failure": None}
            partial_summary = {"incomplete_cells": 1, "successes": None}
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "summarize", return_value=partial_summary), \
                    patch.object(runner, "print", create=True), \
                    patch.object(runner, "run_batch", return_value=(audited_cells, audited_receipt)) as dispatch:
                with self.assertRaisesRegex(ValueError, "no automatic next child"):
                    runner.execute_locked(root, Path("unused.env"), Path("unused.yaml"), "all", handoff_consumer=fixture.consumer)
                self.assertEqual(dispatch.call_count, 1)
                persisted = json.loads((root / "RESULTS.json").read_bytes())
                self.assertEqual(persisted, {"cells": audited_cells, "batches": [audited_receipt]})
                self.assertIsNone(persisted["cells"][0]["survived_horizon"])
                self.assertEqual(json.loads((root / "SUMMARY.json").read_bytes()), partial_summary)
                self.assertTrue((root / "PROVIDER352_COLLECTION_STOP.json").exists())
                fresh = provider352_consumer_fixture(parent_sha)
                with self.assertRaisesRegex(ValueError, "Stopped v2 collection"):
                    runner.execute_locked(root, Path("unused.env"), Path("unused.yaml"), "all", handoff_consumer=fresh.consumer)
                self.assertEqual(dispatch.call_count, 1)
                fresh.health.assert_not_called()

    def test_post_batch_denial_persists_stop_and_blocks_a_fresh_consumer_successor(self):
        candidate = {"models": ["qwen_base", "sp_lense_last8"], "episodes": 2,
                     "batches": [{"models": ["qwen_base"], "seed": 1}, {"models": ["sp_lense_last8"], "seed": 1}],
                     "seeds": [], "protocol": "unit_fixture", "horizon_hours": 48}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.write_json(root / "PLAN.json", candidate)
            parent_sha = runner.sha(root / "PLAN.json")
            fixture = provider352_consumer_fixture(parent_sha)
            def finish_then_revoke(batch, directory, credentials, *, handoff_consumer):
                fixture.records["current-window.json"] = b'{"status":"REVOKED"}'
                return [{"evidence_status": "audited_complete"}], {"batch": directory.name, "exit_code": 0}
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "summarize", return_value={}), \
                    patch.object(runner, "print", create=True), \
                    patch.object(runner, "LOCK", root / "claim.lock"), \
                    patch.object(runner, "run_batch", side_effect=finish_then_revoke) as dispatch:
                with self.assertRaisesRegex(ValueError, "denied"):
                    runner.execute_locked(root, Path("unused.env"), Path("unused.yaml"), "all", handoff_consumer=fixture.consumer)
                self.assertEqual(dispatch.call_count, 1)
                self.assertTrue((root / "PROVIDER352_COLLECTION_STOP.json").exists())
                self.assertEqual(len(json.loads((root / "RESULTS.json").read_bytes())["batches"]), 1)
                fresh = provider352_consumer_fixture(parent_sha)
                with self.assertRaisesRegex(ValueError, "Stopped v2 collection"):
                    runner.execute_locked(root, Path("unused.env"), Path("unused.yaml"), "all", handoff_consumer=fresh.consumer)
                self.assertEqual(dispatch.call_count, 1)
                fresh.health.assert_not_called()


class Provider352CLIParsingTests(unittest.TestCase):
    def test_cli_passes_independent_core_reference_without_reading_a_window(self):
        reference = {"path": "core.json", "sha256": "a" * 64}
        args = ["comparison", "--output", "unused", "--qualified-core-path", reference["path"],
                "--qualified-core-sha256", reference["sha256"]]
        with patch.object(sys, "argv", args), patch.object(runner, "plan", return_value={}) as plan, \
                patch.object(runner, "execute") as execute, patch.object(runner, "print", create=True):
            runner.main()
            self.assertEqual(plan.call_args.kwargs, {"core_binding": reference})
            execute.assert_not_called()
        with patch.object(sys, "argv", args + ["--execute"]), patch.object(runner, "plan") as plan, \
                patch.object(runner, "execute") as execute:
            runner.main()
            self.assertEqual(execute.call_args.kwargs, {"core_binding": reference})
            plan.assert_not_called()


class Provider352ParentActivationTests(unittest.TestCase):
    def setUp(self):
        provider352_isolate_state(self)

    def test_grant_issuance_does_not_change_metadata_plan_bytes_or_call_real_planner(self):
        fixture = provider352_consumer_fixture()
        local = ("qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8",
                 "sp_lense_retrained_neutral", "sp_lense_retrained", "sp_lense_provider352_neutral", "sp_lense_provider352")
        config = {"seeds": (1, 2, 3), "local_models": local, "hosted_models": ("terra", "deepseek"),
                  "world_version": "mock_metadata_only", "schema": "unit", "role": "unit_test",
                  "primary_outcome": "unit", "primary_contrast": [local[-2], local[2]],
                  "same_prompt_adapter_contrasts": [], "comparison_contrasts": []}
        def mock_prospective(seed, *, models, **kwargs):
            return {"models": models, "samples": [{"case_id": f"unit-{seed}-{name}"} for name in models],
                    "horizon_hours": 48, "outer_seconds": 14400,
                    "generation_settings": {name: copy.deepcopy(runner.MODEL_SPECS[name]) for name in models}}
        with patch.object(models, "_bounded_metadata_bytes", side_effect=fixture.read_metadata), \
                patch.object(Path, "read_text", return_value="in-memory unit metadata"), \
                patch.object(runner.yaml, "safe_load", return_value=config), \
                patch.object(runner, "sha", return_value="0" * 64), \
                patch.object(runner, "prospective_plan", side_effect=mock_prospective):
            before = runner.plan(Path("unused-unit.yaml"), core_binding=fixture.core_binding)
            private = fixture.consumer.private_child_context(fixture.pins["expected_game_plan_sha256"])
            with patch.dict(models.os.environ, {models.PROVIDER352_CONTEXT_ENV: private}):
                after = runner.plan(Path("unused-unit.yaml"), core_binding=fixture.core_binding)
        self.assertEqual(json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True))
        self.assertEqual(before["episodes"], 30)
        self.assertNotIn("grant_nonce", json.dumps(before))
        self.assertIsNone(models._PROVIDER352_ACTIVE_CONSUMER)
        fixture.health.assert_not_called()

    def test_public_execute_passes_private_consumer_and_core_before_locked_execution(self):
        fixture = provider352_consumer_fixture()
        private = fixture.consumer.private_child_context(fixture.pins["expected_game_plan_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            destination = repository / ".local/run"
            with patch.dict(models.os.environ, {models.PROVIDER352_CONTEXT_ENV: private}), \
                    patch.object(models, "_bounded_metadata_bytes", side_effect=fixture.read_metadata), \
                    patch.object(runner, "REPOSITORY", repository), \
                    patch.object(runner, "configured_key_names", return_value={"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}), \
                    patch.object(runner, "execute_locked") as locked:
                runner.execute(destination, Path("unused.env"), Path("unused.yaml"), "all", core_binding=fixture.core_binding)
            consumer = locked.call_args.kwargs["handoff_consumer"]
            self.assertEqual(dict(consumer.core_binding), fixture.core_binding)
            self.assertEqual(consumer.requester_chat, fixture.consumer.requester_chat)
            self.assertEqual(consumer.pins["expected_game_plan_sha256"], fixture.pins["expected_game_plan_sha256"])
            self.assertFalse((destination / "RUNNER.lock").exists())
            fixture.health.assert_not_called()

    def test_planning_core_without_grant_cannot_start_even_hosted_phase(self):
        fixture = provider352_consumer_fixture()
        models.initialize_provider352_core(fixture.core_binding, read_bytes=fixture.read_metadata)
        with patch.object(runner, "plan", return_value={"models": ["qwen_base", "terra", "deepseek"]}), \
                patch.object(runner, "run_batch") as dispatch, patch.object(runner, "write_json") as write:
            with self.assertRaisesRegex(ValueError, "NOT_READY"):
                runner.execute_locked(Path("unused-output"), Path("unused.env"), Path("unused.yaml"), "hosted")
            dispatch.assert_not_called()
            write.assert_not_called()
        fixture.health.assert_not_called()


if __name__ == "__main__":
    unittest.main()
