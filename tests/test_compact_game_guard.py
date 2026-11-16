"""CPU fixtures for compact science over an unchanged two-route transport.

All grants, reviews and approvals constructed below are synthetic; no fixture is
an operational admission. Real HELD template bytes are tested as a negative.
"""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from src import compact_game_comparison as compact
from src import company_models as models
from src import hosted_inspect_runner as hosted
from src import independent_state_game as independent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Contract:
    ARMS = ("qwen3.8-27b-base", "sp-20261116-10-bdacaf88")

    @staticmethod
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def same(a, b):
        return canonical(a) == canonical(b)

    @staticmethod
    def ref(value):
        return (type(value) is dict and set(value) == {"path", "sha256"}
                and type(value["path"]) is str and type(value["sha256"]) is str
                and len(value["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in value["sha256"]))

    @staticmethod
    def epoch(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    @staticmethod
    def raw_json(raw, expected):
        if digest(raw) != expected:
            raise ValueError("Fixture bytes changed")
        return json.loads(raw)

    @staticmethod
    def validate_window(value):
        return value.get("fixture_granted") is True and value.get("max_game_requests") == 3072


class CompactGuardTests(unittest.TestCase):
    def setUp(self):
        root = models.PROVIDER352_ROOT / compact.ROOT_TEMPLATE_REF["path"]
        raw = root.read_bytes()
        self.assertEqual(digest(raw), compact.ROOT_TEMPLATE_REF["sha256"])
        self.actual_template = json.loads(raw)
        self.temp = tempfile.TemporaryDirectory(prefix="cg_cpu_", dir=compact.REPOSITORY / ".local")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))
        return {"path": path.relative_to(compact.REPOSITORY).as_posix(), "sha256": digest(path.read_bytes())}

    def fixture(self):
        interface = {name: compact._sha(compact.REPOSITORY / name) for name in compact.RUNTIME_SOURCES}
        export = {compact.EXPORT_SOURCE: compact._sha(compact.REPOSITORY / compact.EXPORT_SOURCE)}
        science = {"interface_runtime_source_sha256": interface, "export_source_sha256": export, "fixture": "SYNTHETIC_NOT_EXECUTABLE"}
        science_ref = self.write("PLAN.json", science)
        template = copy.deepcopy(self.actual_template)
        record = copy.deepcopy(template)
        record.update(schema=template["final_record_schema"], status=compact.GRANTED_STATUS,
                      execution_authorized=True)
        record["scientific_successor"].update(
            science_plan_ref=science_ref,
            source_packet_ref=self.write("source.json", {"interface_runtime_source_sha256": interface, "export_source_sha256": export}),
            paired_world_identity_proof_ref=self.write("pairing.json", {"synthetic": True}))
        for name in record["reviews"]:
            record["reviews"][name] = (dict(path=".local/CPU_SYNTHETIC/root_review.json", sha256="7" * 64)
                if name == "Root_final_exact_amendment_review_ref" else self.write(name + ".json", {"synthetic": True}))
        record["human_approval"].update(approved=True,
            approval_record_ref=self.write("human.json", {"synthetic": True, "approved": True}))
        consumer = object.__new__(compact.CompactIndependentStateConsumer)
        consumer.contract = Contract()
        consumer.contract.MAIN_12CASE_PLAN_REF = copy.deepcopy(template["transport_parent"]["Main_PLAN_external_opaque_ref"])
        consumer.core_binding = {"core_binding": copy.deepcopy(template["transport_parent"]["qualified_Core"])}
        consumer.pins = {"expected_game_plan_sha256": consumer.contract.MAIN_12CASE_PLAN_REF["sha256"],
                         "expected_window_sha256": "8" * 64, "index_path": ".local/CPU_SYNTHETIC/index.json",
                         "expected_index_sha256": "6" * 64, "expected_nonce": "5" * 32}
        consumer.scientific_template_ref = copy.deepcopy(compact.ROOT_TEMPLATE_REF)
        consumer.scientific_amendment_ref = {"path": ".local/CPU_SYNTHETIC/final.json", "sha256": "9" * 64}
        consumer._clock = lambda: Contract.epoch(template["created_utc"]) + 1
        consumer._input_budget = None
        consumer._attempts = 0
        consumer.stop = Mock()
        index_ref = {"path": consumer.pins["index_path"], "sha256": consumer.pins["expected_index_sha256"]}
        window_ref = {"path": ".local/CPU_SYNTHETIC/window.json", "sha256": consumer.pins["expected_window_sha256"]}
        core = {"expected_health": {"pid": template["transport_parent"]["service_pid"],
                "process_created_at": template["transport_parent"]["service_created_epoch"]},
                "two_route_qualification": copy.deepcopy(template["transport_parent"]["qualification"])}
        index = {"core_binding": copy.deepcopy(consumer.core_binding["core_binding"]), "request_window": window_ref}
        window = {"fixture_granted": True, "max_game_requests": 3072,
                  "game_plan_sha256": consumer.pins["expected_game_plan_sha256"],
                  "grant_nonce": consumer.pins["expected_nonce"],
                  "issued_utc": template["created_utc"], "expires_utc": "2026-11-17T17:08:54+00:00"}
        consumer._fixture_window = window

        def reference(ref):
            if ref == consumer.scientific_template_ref:
                value = template
            elif ref == consumer.scientific_amendment_ref:
                value = record
            elif ref == record["reviews"]["Root_final_exact_amendment_review_ref"]:
                value = {"synthetic": True}
            elif ref == consumer.core_binding["core_binding"]:
                value = core
            elif ref == index_ref:
                value = index
            elif ref == window_ref:
                value = window
            else:
                raise ValueError("Unbound synthetic Root reference")
            return canonical(value), copy.deepcopy(value)

        consumer._reference = reference
        return consumer, template, record, science

    def validate(self, consumer, template):
        with patch.object(independent, "assert_candidate_identity"), \
                patch.object(independent, "_CANDIDATE", template["actual_artifact_binding"]):
            return consumer._validated_scientific_amendment()

    def test_actual_held_template_rejected_before_transport_constructor_or_health(self):
        health = Mock(side_effect=AssertionError("No health call"))
        with patch.object(independent.IndependentStateGameConsumer, "__init__") as base:
            with self.assertRaisesRegex(ValueError, "remains HELD"):
                compact.CompactIndependentStateConsumer({}, scientific_amendment_ref=compact.ROOT_TEMPLATE_REF,
                                                        source_binding={}, read_health=health)
        base.assert_not_called()
        health.assert_not_called()

    def test_entrypoints_default_deny_before_health_model_or_child_process(self):
        with patch.object(models, "_read_local_health", side_effect=AssertionError("No health")) as health, \
                patch.object(models, "build_model", side_effect=AssertionError("No model")) as build, \
                patch.object(compact.subprocess, "Popen", side_effect=AssertionError("No child")) as spawn:
            for entrypoint in (compact.execute_child, compact.execute_cohort):
                with self.subTest(entrypoint=entrypoint.__name__), self.assertRaisesRegex(ValueError, "HELD"):
                    entrypoint(self.directory / "NO_EXECUTION", consumer=None)
            self.assertFalse((self.directory / "NO_EXECUTION").exists())
        health.assert_not_called()
        build.assert_not_called()
        spawn.assert_not_called()

    def test_closed_synthetic_final_accepts_and_separates_science_from_transport(self):
        consumer, template, record, _ = self.fixture()
        self.assertEqual(self.validate(consumer, template), record)
        scientific = record["scientific_successor"]["science_plan_ref"]["sha256"]
        transport = consumer.pins["expected_game_plan_sha256"]
        self.assertNotEqual(scientific, transport)
        with patch.object(consumer, "_validated_scientific_amendment", return_value=record):
            consumer.require_plan(scientific)
            with self.assertRaisesRegex(ValueError, "differs"):
                consumer.require_plan(transport)
        consumer.stop.assert_called_once()

    def test_human_reviews_null_science_and_immutable_transport_mutations_reject(self):
        mutations = {
            "human_false": lambda r: r["human_approval"].update(approved=False),
            "human_null": lambda r: r["human_approval"].update(approval_record_ref=None),
            "review_null": lambda r: r["reviews"].update(Main_exact_source_plan_review_ref=None),
            "science_null": lambda r: r["scientific_successor"].update(science_plan_ref=None),
            "quota": lambda r: r["unchanged_protocol"].update(max_Game_requests=3073),
            "clock": lambda r: r["resource_lifecycle"].update(original_whole_seconds=90001),
            "core": lambda r: r["transport_parent"]["qualified_Core"].update(sha256="0" * 64),
            "artifact": lambda r: r["actual_artifact_binding"].update(adapter_sha256="0" * 64),
            "extra_field": lambda r: r.update(extra=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                consumer, template, record, _ = self.fixture()
                mutate(record)
                with self.assertRaises(ValueError):
                    self.validate(consumer, template)

    def test_core_and_transport_helper_must_match_even_when_final_matches_template(self):
        for field in ("core", "plan", "helper"):
            with self.subTest(field=field):
                consumer, template, _, _ = self.fixture()
                if field == "core":
                    consumer.core_binding["core_binding"] = {"path": ".local/wrong.json", "sha256": "0" * 64}
                elif field == "plan":
                    consumer.pins["expected_game_plan_sha256"] = "0" * 64
                else:
                    consumer.contract.MAIN_12CASE_PLAN_REF = {"path": ".local/wrong.json", "sha256": "0" * 64}
                with self.assertRaisesRegex(ValueError, "transport parent/Core"):
                    self.validate(consumer, template)

    def test_matching_template_and_record_cannot_raise_quota_or_reset_original_clock(self):
        for mutate in (lambda r: r["unchanged_protocol"].update(max_Game_requests=3073),
                       lambda r: r["unchanged_protocol"].update(max_case_seconds=14401),
                       lambda r: r["resource_lifecycle"].update(original_whole_seconds=90001),
                       lambda r: r["resource_lifecycle"].update(clock_or_quota_reset_permitted=True)):
            consumer, template, record, _ = self.fixture()
            mutate(template)
            mutate(record)
            with self.assertRaisesRegex(ValueError, "remains HELD"):
                self.validate(consumer, template)

    def test_expired_original_window_rejected_even_with_valid_final_science(self):
        consumer, template, _, _ = self.fixture()
        consumer._fixture_window["expires_utc"] = template["created_utc"]
        with self.assertRaisesRegex(ValueError, "window is absent, changed or expired"):
            self.validate(consumer, template)

    def test_changed_pinned_scientific_bytes_fail_before_admission(self):
        consumer, template, record, _ = self.fixture()
        path = compact.REPOSITORY / record["scientific_successor"]["science_plan_ref"]["path"]
        path.write_bytes(b'{"changed":true}')
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.validate(consumer, template)

    def test_child_view_identity_mutations_never_mark_child_verified(self):
        batch = {"source_sha256": {}, "models": [independent.INDEPENDENT_STATE_IDS[0]], "seed": 123,
                 "world_plan_sha256": "1" * 64, "rules_version": "company_world_long_horizon_v1",
                 "incentive_profile": "high_reward", "generation_settings": {}, "samples": [],
                 "history_mode": compact.HISTORY_MODE, "prompt_protocol": compact.PROTOCOL,
                 "input_view_protocol": compact.PROTOCOL, "user_task_instruction": compact.FULL_HISTORY_USER_TASK,
                 "initial_user_payload_sha256": "2" * 64, "interface_runtime_source_sha256": {"driver": "3" * 64}}
        consumer = object.__new__(compact.CompactIndependentStateConsumer)
        consumer.stop = Mock()
        consumer._validate_game_source_scope = lambda: copy.deepcopy(batch)
        consumer.validate_child_plan(copy.deepcopy(batch))
        self.assertTrue(consumer._child_plan_verified)
        for key in ("history_mode", "prompt_protocol", "input_view_protocol", "user_task_instruction",
                    "initial_user_payload_sha256", "interface_runtime_source_sha256"):
            with self.subTest(key=key):
                child = copy.deepcopy(batch)
                child[key] = None
                with self.assertRaisesRegex(ValueError, "child/source"):
                    consumer.validate_child_plan(child)
                self.assertFalse(consumer._child_plan_verified)

    def test_source_scope_claim_binds_science_and_stop_without_replacing_transport(self):
        consumer, template, record, science = self.fixture()
        seeds = [101, 202, 303]
        parent = {"seeds": seeds, "batches": []}
        for world, model in compact.INDEPENDENT_STATE_EXECUTION_ORDER:
            parent["batches"].append({"models": [model], "seed": seeds[world],
                "world_plan_sha256": str(world + 1) * 64, "source_sha256": {"frozen": "4" * 64},
                "horizon_hours": 48, "max_decisions": 256, "outer_seconds": 14400, "samples": []})
        parent_ref = self.write("parent/PLAN.json", parent)
        template["transport_parent"]["Main_PLAN_external_opaque_ref"] = parent_ref
        record["transport_parent"]["Main_PLAN_external_opaque_ref"] = copy.deepcopy(parent_ref)
        consumer.contract.MAIN_12CASE_PLAN_REF = copy.deepcopy(parent_ref)
        consumer.pins["expected_game_plan_sha256"] = parent_ref["sha256"]
        consumer._fixture_window["game_plan_sha256"] = parent_ref["sha256"]
        protocol_ref = self.write("protocol.json", {"synthetic": True})
        config = {"schema": compact.SCHEMA,
                  "transport_parent_plan": {"path": str(compact.REPOSITORY / parent_ref["path"]), "sha256": parent_ref["sha256"]},
                  "prospective_protocol": {"path": str(compact.REPOSITORY / protocol_ref["path"]),
                                           "sha256": protocol_ref["sha256"]}}
        config_ref = self.write("config.json", config)
        science.update(protocol=compact.SCHEMA, history_mode=compact.HISTORY_MODE,
            prompt_protocol=compact.PROTOCOL, input_view_protocol=compact.PROTOCOL,
            models=list(independent.INDEPENDENT_STATE_IDS), seeds=seeds,
            transport_parent_plan={"path": str(compact.REPOSITORY / parent_ref["path"]), "sha256": parent_ref["sha256"]},
            generation_profile_sha256=models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
            generation_seed=20261108, user_task_instruction=compact.FULL_HISTORY_USER_TASK,
            transport_max_total_native_dispatches=3072, transport_max_additional_native_dispatches=3072,
            configuration_path=str(compact.REPOSITORY / config_ref["path"]), configuration_sha256=config_ref["sha256"],
            prospective_protocol=copy.deepcopy(config["prospective_protocol"]),
            batches=[dict(copy.deepcopy(batch), history_mode=compact.HISTORY_MODE, prompt_protocol=compact.PROTOCOL,
                          input_view_protocol=compact.PROTOCOL, user_task_instruction=compact.FULL_HISTORY_USER_TASK)
                     for batch in parent["batches"]])
        science_ref = self.write("PLAN.json", science)
        record["scientific_successor"]["science_plan_ref"] = science_ref
        science_path = compact.REPOSITORY / science_ref["path"]
        consumer.source_scope = {"plan_path": str(science_path), "batch_index": 0}
        grants = self.directory / "grants"
        grants.mkdir()
        consumer._grant_directory = lambda: grants
        grant = consumer.pins["expected_window_sha256"]
        claim = {"window_sha256": grant, "game_plan_sha256": science_ref["sha256"],
                 "destination": str(science_path.parent), "initial_phase": "all"}
        claim_path = grants / (grant + ".CLAIM.json")
        claim_path.write_bytes(canonical(claim))
        with patch.object(independent, "assert_candidate_identity"), \
                patch.object(independent, "_CANDIDATE", template["actual_artifact_binding"]), \
                patch.object(compact, "source_matches", return_value=True) as check_sources:
            self.assertEqual(consumer._validate_game_source_scope(), science["batches"][0])
            self.assertEqual(check_sources.call_count, 12)
            claim_path.write_bytes(canonical(dict(claim, game_plan_sha256=parent_ref["sha256"])))
            with self.assertRaisesRegex(ValueError, "claim differs"):
                consumer._validate_game_source_scope()
            claim_path.write_bytes(canonical(claim))
            (grants / (grant + ".STOP.json")).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collection stopped"):
                consumer._validate_game_source_scope()
        self.assertEqual(consumer.pins["expected_game_plan_sha256"], parent_ref["sha256"])

    def test_wire_uses_real_shared_route_profile_and_frozen_compact_system(self):
        consumer, _, _, _ = self.fixture()
        consumer._request_parser = lambda: SimpleNamespace(parse_request=lambda raw: json.loads(raw))
        body = {"input_view_protocol": compact.PROTOCOL, "task": compact.FULL_HISTORY_USER_TASK,
                "observation": {}, "action_ledger": []}
        user = canonical(body).decode() + "\nREPLY"
        batch = {"samples": [{"system_prompt": "FORMAT\nICL"}], "response_instruction": "REPLY",
                 "initial_user_payload_sha256": digest(user.encode())}
        consumer._validate_game_source_scope = lambda: copy.deepcopy(batch)
        spec = copy.deepcopy(models.MODEL_SPECS["qwen_base_neutral"])
        profile = hosted.hosted_model_config(spec["model"])
        wire = {key: profile[key] for key in ("temperature", "top_p", "seed")}
        wire.update(profile["extra_body"], model=spec["service_model_id"], max_tokens=8192, stream=True,
                    messages=[{"role": "system", "content": "FORMAT\nICL"}, {"role": "user", "content": user}])
        core = {"base_url": "http://127.0.0.1:8769/v1", "health_url": "http://127.0.0.1:8769/health",
                "per_model": {spec["service_model_id"]: {}}}

        def check(value):
            request = httpx.Request("POST", core["base_url"] + "/chat/completions", content=canonical(value))
            consumer._validate_wire(core, request, spec)

        check(wire)
        self.assertEqual(consumer._input_budget["maximum_prompt_tokens"], 122880)
        for mutation in ("system", "initial", "profile", "protocol"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(wire)
                if mutation == "system":
                    bad["messages"][0]["content"] = "OMITTED ICL"
                elif mutation == "initial":
                    bad["messages"][1]["content"] = user.replace('"observation":{}', '"observation":{"changed":true}')
                elif mutation == "profile":
                    bad["temperature"] = 0
                else:
                    bad["messages"][1]["content"] = user.replace(compact.PROTOCOL, "old_full_history")
                with self.assertRaises(ValueError):
                    check(bad)

    def test_original_ack_deadline_busy_watch_and_no_cache_implementation_inherited(self):
        child = compact.CompactIndependentStateConsumer
        parent = models.ContinuityTransferGameConsumer
        for name in ("install_http_guard", "_wait_for_completed_request", "_read_completion_state",
                     "_remaining_seconds", "_read_health_with_budget", "bind_case_deadline", "_bind_global_deadline"):
            self.assertIs(getattr(child, name), getattr(parent, name), name)
        fake = SimpleNamespace(config=SimpleNamespace(max_retries=5, cache=True),
            api=SimpleNamespace(model_args={}, client=SimpleNamespace(max_retries=5)))
        consumer = SimpleNamespace(install_http_guard=Mock(return_value="guarded"))
        self.assertEqual(models._security500_guard_model(fake, consumer, "condition"), "guarded")
        self.assertEqual((fake.config.max_retries, fake.api.model_args["max_retries"], fake.api.client.max_retries), (0, 0, 0))
        self.assertFalse(fake.config.cache)
        self.assertFalse(fake.should_retry(Exception("do not retry")))


if __name__ == "__main__":
    unittest.main()
