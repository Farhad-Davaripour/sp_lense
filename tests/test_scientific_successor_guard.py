"""Pinned amendment integration, in-memory metadata and mocked child only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import os
import secrets
import unittest
from unittest.mock import Mock, patch

from src import company_models as models
from src import infrastructure_preservation_comparison as runner
from tests.support.local_artifacts import require_local_files


def amendment_fixture(change=None):
    require_local_files(models.SCIENTIFIC_AMENDMENT_TEMPLATE)
    template = json.loads(models.SCIENTIFIC_AMENDMENT_TEMPLATE.read_bytes())
    value = copy.deepcopy(template)
    value.update(status="GRANTED_SCIENTIFIC_SUCCESSOR_UNDER_EXISTING_TRANSPORT", issued_by_root=True,
                 issued_utc="2026-11-16T01:34:00+00:00")
    for name in ("archive_receipt", "consumer_closeout", "preserved_local_STOP"):
        value["retired_comparison"][name] = {"path": ".local/synthetic/" + name + ".json", "sha256": "a" * 64}
    for name in ("scientific_plan_sha256", "protocol_sha256", "client_source_packet_sha256", "cohort_commitment_sha256"):
        value["successor"][name] = "b" * 64
    value["successor"]["history_serializer_source_sha256"] = hashlib.sha256(
        Path(models.__file__).with_name("hosted_inspect_runner.py").read_bytes()).hexdigest()
    value["successor"]["public_amendment_request"] = {"path": ".local/synthetic/request.json", "sha256": "c" * 64}
    value["permissions"]["successor_scientific_model_requests"] = True
    value["additional_human_ICL_authority"] = {"path": ".local/synthetic/icl.json", "sha256": models.SCIENTIFIC_ICL_AUTHORITY_SHA}
    if change:
        change(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ref = {"path": ".local/synthetic/scientific_amendment.json", "sha256": hashlib.sha256(raw).hexdigest()}
    transport = template["transport"]
    pins = {"index_path": transport["index"]["path"], "expected_index_sha256": transport["index"]["sha256"],
        "expected_window_sha256": transport["window"]["sha256"], "expected_nonce": transport["transport_grant_nonce"],
        "expected_game_plan_sha256": transport["transport_plan_sha256"],
        "expected_direct_authority_sha256": "114bf5e1c42e059f048e5a2033aa9042942c565e4a115a3fe757173a4f10fedb"}
    descriptor = {"core_binding": transport["core"], "trusted_review_sha256": "35f732d17eb316437ab2a03effa3b05f3477e4ffbdff75e3e66b95f8511134eb"}
    kwargs = dict(core_binding=descriptor, requester_chat=template["owner_chat"], scientific_amendment_ref=ref,
        read_bytes=lambda path: raw, read_health=Mock(side_effect=AssertionError("No live health")),
        clock=lambda: 1794793000.0)
    return value, pins, kwargs


class ScientificSuccessorTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Native Windows path behavior")
    def test_actual_workspace_short_namespace_can_create_read_claim_and_STOP(self):
        _, pins, kwargs = amendment_fixture()
        consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
        # A fresh full-length test SHA keeps this real workspace check isolated
        # from all actual granted/retired amendment directories.
        consumer.scientific_amendment_ref = {"path": ".local/synthetic/windows_path_only.json",
                                             "sha256": secrets.token_hex(32)}
        directory = consumer._grant_directory()
        expected_base = Path(models.__file__).resolve().parents[1] / ".local/fh_grants"
        self.assertEqual(directory.parent, expected_base)
        claim = directory / (pins["expected_window_sha256"] + ".CLAIM.json")
        stopped = directory / (pins["expected_window_sha256"] + ".STOP.json")
        self.assertLess(len(str(claim)), 260)
        self.assertLess(len(str(stopped)), 260)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with claim.open("x", encoding="utf-8") as stream:
                json.dump({"CPU_native_Windows_path_fixture": True}, stream)
                stream.flush(); os.fsync(stream.fileno())
            self.assertTrue(json.loads(claim.read_bytes())["CPU_native_Windows_path_fixture"])
            consumer.source_scope = {"plan_path": "CPU-path-fixture-only", "batch_index": 0}
            consumer.stop()
            self.assertEqual(json.loads(stopped.read_bytes())["status"], "STOPPED")
        finally:
            # Delete only the exact files and fresh empty test directory we made.
            if claim.exists():
                claim.unlink()
            if stopped.exists():
                stopped.unlink()
            directory.rmdir()

    def test_every_wire_turn_requires_exact_frozen_persistent_system_instruction(self):
        _, pins, kwargs = amendment_fixture()
        consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
        user = "FULL CHRONOLOGICAL TRANSCRIPT v1\nfixture"
        batch = {"initial_user_payload_sha256": hashlib.sha256(user.encode()).hexdigest(),
                 "samples": [{"system_prompt": "frozen system and ICL"}]}
        parser = models.continuity_transfer_request_parser()
        body = {"model": consumer.contract.BASE, "messages": [{"role": "system", "content": "frozen system and ICL"},
                 {"role": "user", "content": user}], "max_tokens": 8192}
        request = Mock(); request.content = json.dumps(body).encode()
        with patch.object(models.ContinuityTransferGameConsumer, "_validate_wire"), \
                patch.object(consumer, "_validate_game_source_scope", return_value=batch):
            consumer._validate_wire({}, request, {})
            consumer._attempts = 1  # Later turns also require the exact system text.
            body["messages"][0]["content"] = "omitted or different ICL"
            request.content = json.dumps(body).encode()
            with self.assertRaisesRegex(ValueError, "system/ICL"):
                consumer._validate_wire({}, request, {})

    def test_child_history_task_protocol_and_initial_wire_hash_join_before_verification(self):
        batch = {"history_mode": "full_chronological_v1", "user_task_instruction": "selected task",
            "prompt_protocol": "company_world_full_chronological_v1", "initial_user_payload_sha256": "1" * 64,
            "world_initial_user_payload_sha256": "2" * 64, "initial_wire_messages_sha256": "3" * 64}
        for key, value in ((None, None), ("history_mode", None), ("user_task_instruction", "wrong"),
                           ("prompt_protocol", "old compact"), ("initial_user_payload_sha256", "9" * 64),
                           ("initial_wire_messages_sha256", "9" * 64)):
            with self.subTest(key=key):
                _, pins, kwargs = amendment_fixture()
                consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
                candidate = copy.deepcopy(batch)
                if key:
                    candidate[key] = value
                with patch.object(consumer, "_validate_game_source_scope", return_value=batch), \
                        patch("src.company_preservation_comparison.validate_batch_manifest"):
                    if key:
                        with self.assertRaises(ValueError):
                            consumer.validate_child_plan(candidate)
                        self.assertFalse(consumer._child_plan_verified)
                    else:
                        consumer.validate_child_plan(candidate)
                        self.assertTrue(consumer._child_plan_verified)

    def test_positive_separate_plan_and_namespace_preserve_transport_and_old_stop(self):
        value, pins, kwargs = amendment_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(models, "CONTINUITY_TRANSFER_GRANT_DIR", Path(tmp)):
            old = Path(tmp) / (pins["expected_window_sha256"] + ".STOP.json")
            old.write_text("retired comparison STOP", encoding="utf-8")
            consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
            consumer.require_plan(value["successor"]["scientific_plan_sha256"])
            self.assertEqual(runner._two_route_grants(consumer), consumer._grant_directory())
            self.assertNotEqual(consumer._grant_directory(), Path(tmp))
            child = json.loads(consumer.private_child_context(value["successor"]["scientific_plan_sha256"],
                source_scope={"plan_path": str(Path(tmp) / "PLAN.json"), "batch_index": 0}))
            self.assertEqual(child["pins"], pins)
            self.assertEqual(child["pins"]["expected_game_plan_sha256"], value["transport"]["transport_plan_sha256"])
            self.assertEqual(set(child), {"pins", "requester_chat", "core_binding", "source_scope"})
            self.assertEqual(old.read_text(encoding="utf-8"), "retired comparison STOP")

    def test_held_or_changed_transport_limits_prompt_or_authority_refuse(self):
        changes = (
            lambda v: v.update(status="HELD_PENDING_FROZEN_SUCCESSOR_AND_ARCHIVE"),
            lambda v: v["transport"].update(max_total_native_dispatches=3072),
            lambda v: v["limits"].update(max_additional_native_dispatches=1536),
            lambda v: v["successor"].update(shared_user_prompt="different"),
            lambda v: v["additional_human_ICL_authority"].update(sha256="9" * 64),
            lambda v: v["permissions"].update(model_reload_or_new_endpoint=True),
        )
        for change in changes:
            with self.subTest(change=change):
                _, pins, kwargs = amendment_fixture(change)
                with self.assertRaises(ValueError):
                    models.ScientificSuccessorGameConsumer(pins, **kwargs)

    def test_scientific_checks_surround_unchanged_transport_and_post_ACK(self):
        value, pins, kwargs = amendment_fixture()
        consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
        with patch.object(models.ContinuityTransferGameConsumer, "_validated_request", return_value="physical") as transport, \
                patch.object(consumer, "_validated_scientific_amendment", wraps=consumer._validated_scientific_amendment) as science:
            self.assertEqual(consumer._validated_request(consumer.contract.BASE), "physical")
            self.assertEqual(science.call_count, 2)
            transport.assert_called_once_with(consumer.contract.BASE)
        consumer._last_completed_evidence = {"dispatch_record": {"game_dispatch_ordinal": 29}}
        with patch.object(models.ContinuityTransferGameConsumer, "_validate_completed_request", return_value="ACK"):
            self.assertEqual(consumer._validate_completed_request(consumer.contract.BASE, "1" * 32, "2" * 64), "ACK")
            consumer._last_completed_evidence["dispatch_record"]["game_dispatch_ordinal"] = 28
            with self.assertRaises(ValueError):
                consumer._validate_completed_request(consumer.contract.BASE, "1" * 32, "2" * 64)

    def test_parent_child_dryrun_forwards_new_mode_and_late_overlay_without_process(self):
        value, pins, kwargs = amendment_fixture()
        consumer = models.ScientificSuccessorGameConsumer(pins, **kwargs)
        consumer.remaining_global_seconds = Mock(return_value=1000)
        batch = {"rules_version": "company_world_long_horizon_v1", "seed": 101,
            "models": ["qwen_base_operational_icl"], "horizon_hours": 48, "outer_seconds": 14400,
            "history_mode": "full_chronological_v1", "user_task_instruction": value["successor"]["shared_user_prompt"]}
        process = Mock(); process.wait.return_value = 0; process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "batch_01"
            with patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
                    patch.object(runner, "audit_batch", return_value=([], {})):
                runner.run_batch(batch, directory, None, security500_consumer=consumer)
            command = launch.call_args.args[0]
            env = launch.call_args.kwargs["env"]
            self.assertIn("--history-mode", command)
            self.assertEqual(command[command.index("--user-task-instruction") + 1], value["successor"]["shared_user_prompt"])
            self.assertEqual(json.loads(env[models.SCIENTIFIC_AMENDMENT_ENV]), consumer.scientific_amendment_ref)
            context = json.loads(env[models.CONTINUITY_TRANSFER_CONTEXT_ENV])
            self.assertEqual(context["pins"], pins)
            self.assertEqual(context["source_scope"]["plan_path"], str((Path(tmp) / "PLAN.json").resolve()))
            process.wait.assert_called_once_with(timeout=1005)


if __name__ == "__main__":
    unittest.main()
