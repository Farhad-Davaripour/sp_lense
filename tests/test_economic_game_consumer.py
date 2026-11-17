"""In-memory synthetic admission; no checkpoint, service, network or study case.

The tiny helper below is a test double, never an operational Root contract.
Native transport/ACK implementations are asserted unchanged, not qualified here.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import company_models as models
from src import economic_game_comparison as comparison
from src import economic_game_consumer as new
from src import economic_identity_registry as registry
from src import hosted_inspect_runner as hosted
from src import independent_state_game as old
from tests.test_economic_identity_registry import fixture as artifact_fixture


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


HELPER = '''# SYNTHETIC TEST DOUBLE ONLY; NEVER PUBLISH OR USE FOR A MODEL
import hashlib, json, re
from datetime import datetime
def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
def ref(value):
    return type(value) is dict and set(value)=={"path","sha256"} and is_sha(value["sha256"]) and type(value["path"]) is str and not any(x in value["path"] for x in ("\\\\", ":", ".."))
def need(value, message):
    if not value: raise ValueError(message)
def same(left, right): return type(left) is type(right) and left == right
def epoch(value): return datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()
def raw_json(raw, sha):
    need(hashlib.sha256(raw).hexdigest()==sha,"changed fixture")
    return json.loads(raw)
def require_new_static_binding():
    need(ROOT_STATIC_BINDING_STATUS=="SYNTHETIC_TEST_BOUND","HELD")
def validate_core(core, trusted):
    need(core["synthetic_only"] is True and trusted=="e"*64,"invalid test Core")
def validate_window(window):
    return window.get("status")=="SYNTHETIC_TEST_GRANTED" and window.get("max_game_requests")==3072
def validate_per_request_metadata(*args, **kwargs):
    return {"metadata_request_allowed": True, "synthetic_only": True}
'''


class EconomicConsumerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "Root"
        self.main = Path(tmp.name) / "Main"
        self.root.mkdir(); self.main.mkdir()
        for owner, name, value in ((new, "REPOSITORY", self.main), (models, "PROVIDER352_ROOT", self.root),
                (registry, "_CANDIDATE", None), (registry, "_SPECS", None)):
            handle = patch.object(owner, name, value); handle.start(); self.addCleanup(handle.stop)
        for name in ("_CONTINUITY_TRANSFER_CORE_REF", "_SECURITY500_CORE_REF", "_PROVIDER352_CORE_REF"):
            handle = patch.object(models, name, None); handle.start(); self.addCleanup(handle.stop)
        for name in ("_CORE_REF", "_ACTIVE"):
            handle = patch.object(old, name, None); handle.start(); self.addCleanup(handle.stop)
        for handle in (patch.dict(models.MODEL_SPECS, copy.deepcopy(models.MODEL_SPECS), clear=True),
                       patch.object(hosted, "LOCAL_QWEN_MODELS", set(hosted.LOCAL_QWEN_MODELS)),
                       patch.dict(new.os.environ, {}, clear=True)):
            handle.start(); self.addCleanup(handle.stop)
        self.records = {}
        self.health = Mock(return_value={"synthetic_only": True})

    def reference(self, relative, value, *, main=False, raw=False):
        data = value if raw else encoded(value)
        self.records[(self.main if main else self.root) / relative] = data
        return {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}

    def reader(self, path):
        return self.records[Path(path)]

    def prepare(self):
        candidate, _, _, data = artifact_fixture()
        for reference in (candidate["artifact_ready"], candidate["completion_audit"]):
            self.records[self.root / reference["path"]] = data[Path(reference["path"]).name]
        registry.register_candidate(candidate, read_bytes=self.reader)
        protocol = self.reference(".local/SYNTHETIC/protocol.yaml", b"synthetic_only: true\n", main=True, raw=True)
        config = self.reference(".local/SYNTHETIC/config.json", {"schema": new.CONFIG_SCHEMA,
            "prospective_protocol": protocol}, main=True)
        # Fixed small integer fixtures are synthetic; no evaluation or model runs.
        with patch.object(comparison, "read_main_bytes", side_effect=lambda ref:
                (self.main / ref["path"], self.records[self.main / ref["path"]])):
            plan = comparison.build_cohort([1, 2, 3], excluded_seeds={10, 11}, configuration_ref=config)
        plan_ref = self.reference(".local/SYNTHETIC/PLAN.json", plan, main=True)
        audit = candidate["completion_audit"]
        review = self.reference(".local/SYNTHETIC/root-review.json", {"status": "PASS_SYNTHETIC_ONLY"})
        ready = candidate["artifact_ready"]
        certificate = self.reference(".local/SYNTHETIC/certificate.json", {"synthetic_only": True})
        core = {"synthetic_only": True, "per_model": {candidate["expected_ft_id"]: {
            "adapter_sha256": candidate["adapter_sha256"], "adapter_config_sha256": candidate["adapter_config_sha256"]},
            "qwen3.8-27b-base": {}}, "base_url": "http://127.0.0.1:8769/v1",
            "health_url": "http://127.0.0.1:8769/health", "readiness_receipt": ready,
            "two_route_qualification": certificate,
            "artifacts": {"completion_audit": audit, "completion_auditor_review": review}}
        core_binding = {"core_binding": self.reference(".local/SYNTHETIC/core.json", core), "trusted_review_sha256": "e" * 64}
        window = {"status": "SYNTHETIC_TEST_GRANTED", "core_binding": core_binding["core_binding"],
            "game_plan_sha256": plan_ref["sha256"], "grant_nonce": "c" * 32, "max_game_requests": 3072,
            "issued_utc": "2026-11-16T16:00:00Z", "expires_utc": "2026-11-17T16:00:00Z",
            "original_direct_authority": {"path": ".local/SYNTHETIC/authority.json", "sha256": "d" * 64}}
        window_ref = self.reference(".local/SYNTHETIC/window.json", window)
        index = {"core_binding": core_binding["core_binding"], "request_window": window_ref}
        index_ref = self.reference(".local/SYNTHETIC/index.json", index)
        pins = {"index_path": index_ref["path"], "expected_index_sha256": index_ref["sha256"],
            "expected_window_sha256": window_ref["sha256"], "expected_nonce": "c" * 32,
            "expected_game_plan_sha256": plan_ref["sha256"], "expected_direct_authority_sha256": "d" * 64}
        constants = {"ECONOMIC_ADMISSION_SCHEMA": new.ADMISSION_SCHEMA, "BASE": "qwen3.8-27b-base",
            "FT": candidate["expected_ft_id"], "ACTUAL_FIT_UPDATES": 612, "ACTUAL_TEACHER_DEV_ROWS": 42,
            "MAX_GAME_REQUESTS": 3072, "MAX_GAME_WINDOW_SECONDS": 90000,
            "FINAL_ADAPTER_SHA": candidate["adapter_sha256"], "FINAL_CONFIG_SHA": candidate["adapter_config_sha256"],
            "BASE_SHA": candidate["base_manifest_sha256"], "READY_ARTIFACT_REF": ready,
            "COMPLETION_AUDIT_SHA": audit["sha256"], "GAME_CHAT": "SYNTHETIC_FUTURE_GAME",
            "MAIN_12CASE_PLAN_REF": plan_ref, "ROOT_STATIC_BINDING_STATUS": "SYNTHETIC_TEST_BOUND"}
        helper = HELPER + "\n" + "\n".join(f"{key}={value!r}" for key, value in constants.items()) + "\nARMS=(BASE,FT)\n"
        parser = ("import json\ndef parse_request(raw): return json.loads(raw)\n"
            "def _require_static_route_binding(): pass\n" + f"LAST={candidate['expected_ft_id']!r}\n"
            f"ARMS=('qwen3.8-27b-base',LAST)\nREADY_ARTIFACT_REF={ready!r}\n"
            f"ECONOMIC_INPUT_PROTOCOL={new.economic.PROTOCOL!r}\n")
        source = {"helper": self.reference(".local/SYNTHETIC/helper.py", helper.encode(), raw=True),
            "parser": self.reference(".local/SYNTHETIC/parser.py", parser.encode(), raw=True), "expected_ft_id": candidate["expected_ft_id"]}
        packet = {"schema": new.SOURCE_PACKET_SCHEMA,
            "interface_runtime_source_sha256": comparison._digests(comparison.RUNTIME_SOURCES),
            "export_source_sha256": comparison._digests(comparison.EXPORT_SOURCES)}
        record = {"schema": new.ADMISSION_SCHEMA, "status": new.ADMITTED_STATUS, "issuer": "Root",
            "issued_by_root": True, "owner_chat": constants["GAME_CHAT"], "issued_utc": window["issued_utc"],
            "actual_artifact_binding": candidate, "source_binding": source, "core_binding": core_binding, "pins": pins,
            "main_plan_ref": plan_ref, "source_packet_ref": self.reference(".local/SYNTHETIC/packet.json", packet, main=True),
            "excluded_seeds": [10, 11], "reviews": {"Root_transport_ref": review,
                "Main_source_ref": self.reference(".local/SYNTHETIC/main-review.json", {"status": "PASS_SYNTHETIC_ONLY"}, main=True)},
            "transport": copy.deepcopy(new.TRANSPORT)}
        admission = self.reference(".local/SYNTHETIC/admission.json", record)
        kwargs = {"pins": pins, "core_binding": core_binding, "requester_chat": constants["GAME_CHAT"],
            "source_binding": source, "admission_ref": admission, "read_bytes": self.reader, "read_health": self.health,
            "clock": lambda: datetime(2026, 11, 16, 16, 1, tzinfo=timezone.utc).timestamp(), "monotonic": lambda: 1.0}
        return kwargs, record, plan, core

    def consumer(self):
        kwargs, record, plan, core = self.prepare()
        return new.EconomicGameConsumer(**kwargs), kwargs, record, plan, core

    def claim(self, consumer):
        path, _ = consumer._main_reference(consumer._scientific_binding["main_plan_ref"])
        path.parent.mkdir(parents=True, exist_ok=True)
        consumer.source_scope = {"plan_path": str(path), "batch_index": 0}
        directory = consumer._grant_directory(); directory.mkdir(parents=True)
        (directory / (consumer.pins["expected_window_sha256"] + ".CLAIM.json")).write_bytes(encoded({
            "window_sha256": consumer.pins["expected_window_sha256"],
            "game_plan_sha256": consumer.pins["expected_game_plan_sha256"], "destination": str(path.parent), "initial_phase": "all"}))
        return path

    def test_true_admitted_constructor_and_native_transport_reuse(self):
        with patch.object(old, "initialize_independent_state_source", side_effect=AssertionError("historical singleton")), \
                patch.object(old.IndependentStateGameConsumer, "__init__", side_effect=AssertionError("old constructor")):
            consumer, _, record, _, _ = self.consumer()
        self.assertIsInstance(consumer, old.IndependentStateGameConsumer)
        self.assertEqual(consumer._validated_scientific_amendment(), record)
        self.assertEqual(consumer.excluded_seeds, frozenset({10, 11}))
        self.assertIsNone(old._CORE_REF)
        self.health.assert_not_called()
        for name in ("install_http_guard", "_wait_for_completed_request", "_validate_completed_request",
                "bind_case_deadline", "remaining_global_seconds", "_remaining_seconds", "_read_completion_state"):
            self.assertIs(getattr(new.EconomicGameConsumer, name), getattr(models.ContinuityTransferGameConsumer, name))
        self.assertIn("economic_grants", str(consumer._grant_directory()))
        self.assertNotEqual(consumer._grant_directory(), models.CONTINUITY_TRANSFER_GRANT_DIR)

    def test_true_held_missing_and_wrong_helper_constructors_fail_before_health(self):
        kwargs, record, _, _ = self.prepare()
        for status in ("HELD", "PREPARED_ONLY", False):
            changed = copy.deepcopy(record); changed["status"] = status
            attempt = dict(kwargs, admission_ref=self.reference(".local/SYNTHETIC/held.json", changed))
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "HELD"):
                new.EconomicGameConsumer(**attempt)
        attempt = dict(kwargs, admission_ref={"path": ".local/SYNTHETIC/missing.json", "sha256": "a" * 64})
        with self.assertRaises((KeyError, ValueError, FileNotFoundError)): new.EconomicGameConsumer(**attempt)
        changed = copy.deepcopy(record)
        helper = self.records[self.root / kwargs["source_binding"]["helper"]["path"]] + b"\nACTUAL_FIT_UPDATES=168\n"
        changed["source_binding"]["helper"] = self.reference(".local/SYNTHETIC/wrong.py", helper, raw=True)
        attempt = dict(kwargs, source_binding=changed["source_binding"], admission_ref=self.reference(".local/SYNTHETIC/wrong.json", changed))
        with self.assertRaisesRegex(ValueError, "actual612"): new.EconomicGameConsumer(**attempt)
        self.health.assert_not_called()

    def test_main_refs_reject_drive_absolute_traversal_backslash_and_aliases(self):
        for path in ("C:/Main/.local/PLAN.json", "C:\\Main\\.local\\PLAN.json", "/.local/PLAN.json",
                ".local/../PLAN.json", ".local//PLAN.json", ".local/./PLAN.json", "PLAN.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                new._pinned_bytes({"path": path, "sha256": "a" * 64}, self.main, self.reader, local=True)

    def test_new_child_context_reconstructs_only_new_registry_and_checks_all_cells(self):
        consumer, kwargs, record, _, _ = self.consumer()
        plan_path = self.main / record["main_plan_ref"]["path"]
        raw = consumer.private_child_context(record["main_plan_ref"]["sha256"], source_scope={"plan_path": str(plan_path), "batch_index": 0})
        self.assertEqual(set(json.loads(raw)), {"pins", "requester_chat", "core_binding", "source_scope", "source_binding", "admission_ref"})
        with patch.object(registry, "_CANDIDATE", None), patch.object(registry, "_SPECS", None), \
                patch.object(old, "register_independent_state_candidate", side_effect=AssertionError("old candidate")):
            child = new.consumer_from_context(raw, **{key: kwargs[key] for key in ("read_bytes", "read_health", "clock", "monotonic")})
            self.assertEqual(child._scientific_binding, record)
            self.assertEqual(child.source_scope["batch_index"], 0)
        with patch.dict(new.os.environ, {old.INDEPENDENT_STATE_CONTEXT_ENV: "old"}), self.assertRaises(ValueError):
            new.consumer_from_context(raw)
        self.health.assert_not_called()

    def test_claim_source_child_manifest_and_stop_are_isolated(self):
        consumer, _, _, plan, _ = self.consumer()
        path = self.claim(consumer)
        self.assertEqual(consumer._validate_game_source_scope(), plan["batches"][0])
        consumer.validate_child_plan(copy.deepcopy(plan["batches"][0]))
        self.assertTrue(consumer._child_plan_verified)
        consumer.source_scope["batch_index"] = True
        with self.assertRaises(ValueError): consumer._validate_game_source_scope()
        consumer.source_scope["batch_index"] = 0
        consumer.stop()
        with self.assertRaisesRegex(ValueError, "stopped"): consumer._validate_game_source_scope()
        self.assertTrue((consumer._grant_directory() / (consumer.pins["expected_window_sha256"] + ".STOP.json")).is_file())
        self.assertEqual(path.name, "PLAN.json")

    def test_source_bytes_and_window_tampering_fail_without_health(self):
        consumer, _, _, _, _ = self.consumer()
        path = self.root / consumer.source_binding["parser"]["path"]
        self.records[path] += b"\n# mutation\n"
        with self.assertRaisesRegex(ValueError, "bytes changed"): consumer.validate_request(consumer.contract.BASE)
        self.assertTrue(consumer._denied)
        self.health.assert_not_called()

    def test_true_native_request_and_economic_carrier_checks(self):
        consumer, _, _, plan, core = self.consumer()
        self.claim(consumer)
        batch = plan["batches"][0]
        role = batch["models"][0]
        spec = models.MODEL_SPECS[role]
        profile = hosted.hosted_model_config(spec["model"])
        world = comparison.evaluation.episode(batch["samples"][0])
        user = comparison.economic.EconomicGameEpisode(world, user_task_instruction=batch["user_task_instruction"]).prompt() + "\n" + batch["response_instruction"]
        body = {key: profile[key] for key in ("temperature", "top_p", "seed")}
        body.update(profile["extra_body"], model=spec["service_model_id"], max_tokens=8192, stream=True,
            messages=[{"role": "system", "content": batch["samples"][0]["system_prompt"]}, {"role": "user", "content": user}])
        request = SimpleNamespace(method="POST", url=core["base_url"] + "/chat/completions", content=encoded(body))
        consumer._validate_wire(core, request, spec)
        self.assertEqual(consumer._input_budget["maximum_prompt_tokens"], 122880)
        self.assertEqual(consumer._economic_case_binding, {"case_id": batch["samples"][0]["case_id"], "decision": 0, "attempt": 1})
        directory = Path(consumer.source_scope["plan_path"]).parent / "batch_01"
        directory.mkdir()
        digest = hashlib.sha256(json.dumps(body["messages"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        identity = {"request_id": "f" * 32, "messages_sha256": digest}
        consumer._record_request_evidence({"event": "HTTP_ATTEMPT", "attempt": 7, **identity})
        # Completion binds the recorded request, never whichever decision is current.
        consumer._economic_case_binding = {"case_id": "different", "decision": 8, "attempt": 1}
        consumer._record_request_evidence({"event": "POST_REQUEST_VALIDATED", **identity})
        rows = [json.loads(line) for line in (directory / "REQUEST_GUARD_EVIDENCE.jsonl").read_text().splitlines()]
        expected = {"case_id": batch["samples"][0]["case_id"], "decision": 0, "attempt": 1}
        self.assertEqual([row["generation_identity"] for row in rows], [expected, expected])
        self.assertEqual(rows[0]["attempt"], 7)
        with self.assertRaises(ValueError):
            consumer._record_request_evidence({"event": "POST_REQUEST_VALIDATED", **identity, "messages_sha256": "a" * 64})
        self.assertEqual(consumer.validate_request(spec["service_model_id"])["synthetic_only"], True)
        payload = json.loads(user[:-len("\n" + batch["response_instruction"])])
        consumer._attempts = 1
        for value in (None, True, -1, 2, 1.0, "1"):
            changed_payload = copy.deepcopy(payload)
            changed_payload["observation"]["decision_count"] = value
            changed_payload["economic_summary"] = comparison.economic_summary(changed_payload)
            body["messages"][1]["content"] = json.dumps(changed_payload) + "\n" + batch["response_instruction"]
            changed = SimpleNamespace(method="POST", url=core["base_url"] + "/chat/completions", content=encoded(body))
            with self.subTest(decision=value), self.assertRaisesRegex(ValueError, "current case decision"):
                consumer._validate_wire(core, changed, spec)
        body["messages"][1]["content"] = "wrong"
        changed = SimpleNamespace(method="POST", url=core["base_url"] + "/chat/completions", content=encoded(body))
        with self.assertRaises(ValueError): consumer._validate_wire(core, changed, spec)


if __name__ == "__main__": unittest.main()
