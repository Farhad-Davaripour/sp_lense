"""Choice visibility and exact provider routing; never call an inference API."""
from pathlib import Path
import hashlib
import json
import copy
import io
import tempfile
import asyncio
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from unittest import SkipTest, TestCase
from unittest.mock import Mock, patch

from src import company_models as models
from src.hosted_inspect_runner import hosted_model_config

QWEN_ENV = {"QWEN_API_KEY": "fixture", "QWEN_BASE_URL": models.qwen_cloud.BASE_URL,
            "QWEN_MODEL": "qwen3.8-max"}
CURRENT_LOCAL_MODELS = ("qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8")
PROVIDER352_MODELS = ("sp_lense_provider352_neutral", "sp_lense_provider352")


def write_credentials(path):
    path.write_text("OPENAI_API_KEY=fixture\nDEEPSEEK_API_KEY=fixture\n" +
                    "\n".join(f"{key}={value}" for key, value in QWEN_ENV.items()) + "\n")


def new_binding_fixture(root, port=8871):
    receipt = root / "new_readiness.json"
    receipt.write_text(json.dumps({"adapter_sha256": models.LAST8_SHA,
                                   "adapter_config_sha256": models.LAST8_CONFIG_SHA}))
    binding = {"adapter_sha256": models.LAST8_SHA,
        "base_url": f"http://127.0.0.1:{port}/v1", "health_url": f"http://127.0.0.1:{port}/health",
        "readiness_receipt_path": str(receipt),
        "readiness_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "expected_health": {"ready": True, "adapter_sha256": models.LAST8_SHA,
            "adapter_config_sha256": models.LAST8_CONFIG_SHA,
            "base_revision": models.LOCAL_BASE_REVISION, "base_manifest_sha256": models.LOCAL_BASE_MANIFEST_SHA256,
            "model_ids": ["qwen3.8-27b-base", models.LAST8_SERVICE_MODEL_ID],
            "max_concurrency": 1, "resident_base_count": 1,
            "limits": {"max_output_tokens": 8192, "context_total_tokens": models.LAST8_CONTEXT_TOKENS,
                       "max_context_tokens_including_output": models.LAST8_CONTEXT_TOKENS}}}
    path = root / "new_binding.json"
    path.write_text(json.dumps(binding))
    return path, binding


def provider352_consumer_fixture(parent_plan_sha256=None, requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4"):
    """Root's synthetic CPU fixture, with real byte hashes; never study evidence."""
    if not models.PROVIDER352_HANDOFF_PATH.parent.exists():
        raise SkipTest("Local Root handoff integration packet is not installed in this checkout")
    path = models.PROVIDER352_HANDOFF_PATH.with_name("test_handoff_contract.py")
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == "7e4f8f7c3f44e2e08e4f7bb62b5067451beef3753c96407e71758259e4cd4dbd"
    module = ModuleType("root_synthetic_handoff_fixture")
    module.__file__ = str(path)
    prior_path, prior_contract = list(sys.path), sys.modules.get("handoff_contract")
    sys.modules.pop("handoff_contract", None)
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        sys.path[:] = prior_path
        if prior_contract is None:
            sys.modules.pop("handoff_contract", None)
        else:
            sys.modules["handoff_contract"] = prior_contract
    core, window = module.fixture()
    core["model_display_names"][models.PROVIDER352_SERVICE_MODEL_ID] = "SP Lens -2026-11-14 23:39 ABT"
    core["per_model"][models.PROVIDER352_SERVICE_MODEL_ID].update(
        adapter_sha256=models.PROVIDER352_TECHNICAL_PINS["adapter"],
        adapter_config_sha256=models.PROVIDER352_TECHNICAL_PINS["adapter_config"])
    for role, digest in models.PROVIDER352_TECHNICAL_PINS.items():
        core["provider_artifacts"][role]["sha256"] = digest
    core["expected_health"]["per_model"] = copy.deepcopy(core["per_model"])
    if parent_plan_sha256 is not None:
        window["game_plan_sha256"] = parent_plan_sha256
    receipt_raw = module.raw(core["expected_health"])
    qualification_raw = module.raw(module.identity_metadata(core)["four_route_qualification"])
    core["readiness_receipt_sha256"] = module.sha(receipt_raw)
    core["four_route_request_qualification"]["sha256"] = module.sha(qualification_raw)
    core_raw = module.raw(core)
    window["core_binding"]["sha256"] = module.sha(core_raw)
    window["bound_proofs"] = module.h.linked_proofs(core)
    window_raw = module.raw(window)
    index = {"schema": module.h.INDEX_SCHEMA, "status": "READY_FOR_GAME_BOUND_WINDOW", "ready": True,
             "core_binding": module.ref("core.json", module.sha(core_raw)),
             "request_window": module.ref("current-window.json", module.sha(window_raw))}
    index_raw = module.raw(index)
    records = {"index.json": index_raw, "core.json": core_raw, "current-window.json": window_raw,
               core["readiness_receipt_path"]: receipt_raw,
               core["four_route_request_qualification"]["path"]: qualification_raw}
    chain_path = core["provider_artifacts"]["completion_auditor_review"]["path"]
    records[chain_path] = (models.PROVIDER352_ROOT / chain_path).read_bytes()
    chain = json.loads(records[chain_path])
    for role in ("full_review_v1", "focused_review_v2", "narrow_root_source_proof_peer_v3", "game_source_proof_review_v3"):
        relative = chain[role]["path"]
        records[relative] = (models.PROVIDER352_ROOT / relative).read_bytes()
    pins = {"index_path": "index.json", "expected_index_sha256": module.sha(index_raw),
            "expected_window_sha256": module.sha(window_raw), "expected_nonce": window["grant_nonce"],
            "expected_game_plan_sha256": window["game_plan_sha256"],
            "expected_direct_authority_sha256": window["original_direct_authority"]["sha256"]}
    reads = []
    def read_metadata(path):
        relative = path.relative_to(models.PROVIDER352_ROOT.resolve()).as_posix()
        reads.append(relative)
        return records[relative]
    health = Mock(side_effect=lambda binding: copy.deepcopy(binding["expected_health"]))
    clock = Mock(return_value=datetime(2026, 11, 15, 2, 30, tzinfo=timezone.utc).timestamp())
    core_binding = dict(index["core_binding"])
    consumer = models.Provider352GameConsumer(pins, core_binding=core_binding, requester_chat=requester_chat,
                                             read_bytes=read_metadata, read_health=health, clock=clock)
    return SimpleNamespace(consumer=consumer, core=core, window=window, index=index, pins=pins,
                           core_binding=core_binding, read_metadata=read_metadata,
                           records=records, reads=reads, health=health, clock=clock, raw=module.raw)


def provider352_isolate_state(test):
    for name in ("_PROVIDER352_CORE_REF", "_PROVIDER352_CORE_RAW", "_PROVIDER352_CORE_SPECS",
                 "_PROVIDER352_ACTIVE_CONTEXT", "_PROVIDER352_ACTIVE_CONSUMER"):
        item = patch.object(models, name, None)
        item.start()
        test.addCleanup(item.stop)
    for name in PROVIDER352_MODELS:
        item = patch.dict(models.MODEL_SPECS[name], copy.deepcopy(models.MODEL_SPECS[name]), clear=True)
        item.start()
        test.addCleanup(item.stop)
    item = patch.dict(models.os.environ)
    item.start()
    models.os.environ.pop(models.PROVIDER352_CONTEXT_ENV, None)
    test.addCleanup(item.stop)


def provider352_fake_http_model(model_id):
    spec = models.MODEL_SPECS[model_id]
    client = SimpleNamespace(event_hooks={"request": [], "response": [object()]},
                             transport=object(), retries=2, cache_marker=object())
    class FakeModel:
        api = SimpleNamespace(client=SimpleNamespace(_client=client))
        config = SimpleNamespace(max_retries=0, max_tokens=8192, max_connections=1)
        def __str__(self):
            return spec["model"]
    return FakeModel(), client


def provider352_fake_wire(fixture, model_id):
    spec = models.MODEL_SPECS[model_id]
    profile = hosted_model_config(spec["model"])
    body = {key: profile[key] for key in ("temperature", "top_p", "seed")}
    body.update(profile["extra_body"], model=spec["service_model_id"], max_tokens=8192,
                stream=True, messages=[{"role": "user", "content": "synthetic transport fixture"}])
    return SimpleNamespace(method="POST", url=fixture.core["base_url"] + "/chat/completions",
                           content=json.dumps(body).encode())


class CompanyModelTests(TestCase):
    def test_full_catalog_does_not_shrink_to_the_operator_test_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / ".env"
            write_credentials(credentials)
            with patch.object(models, "current_local_binding", return_value={}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.dict(models.os.environ, QWEN_ENV), \
                    patch.object(models.urllib.request, "urlopen", side_effect=AssertionError("No readiness probe for catalog")):
                choices = models.model_choices(credentials)
            self.assertEqual(len(choices), 17)
            self.assertEqual({m["id"] for m in choices if m["available"]},
                             {key for key, spec in models.MODEL_SPECS.items()
                              if not spec.get("retired") and not spec.get("requires_scientific_amendment") and spec.get("service_model_id") not in
                              {models.RETRAINED_SERVICE_MODEL_ID, models.PROVIDER352_SERVICE_MODEL_ID,
                               models.SECURITY500_SERVICE_MODEL_ID, models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID}})
            self.assertEqual([m["id"] for m in choices if m["default_selected"]], ["deepseek"])
            self.assertFalse(choices[-1]["available"])
            self.assertIn("not verified", choices[-1]["status"])

    def test_neutral_and_instruction_arms_bind_the_same_exact_weights(self):
        for neutral, instructed in (("qwen_base_neutral", "qwen_base"), ("sp_lense_last8_neutral", "sp_lense_last8")):
            self.assertEqual(models.MODEL_SPECS[neutral]["model"], models.MODEL_SPECS[instructed]["model"])
            self.assertEqual(models.MODEL_SPECS[neutral]["prompt_condition"], "baseline")
            self.assertEqual(models.MODEL_SPECS[instructed]["prompt_condition"], "security_maximization")
        self.assertTrue(models.MODEL_SPECS["sp_lense_last8"]["model"].endswith(models.LAST8_SERVICE_MODEL_ID))
        self.assertEqual(models.MODEL_SPECS["sp_lense"]["adapter_sha256"], models.LAST_SHA)
        self.assertTrue(models.MODEL_SPECS["sp_lense"]["retired"])
        self.assertEqual(models.MODEL_SPECS["qwen_base"]["model"], "openai-api/research4-qwen/qwen3.8-27b-base")
        self.assertTrue(models.MODEL_SPECS["qwen_base"]["local_qwen"])
        self.assertEqual(models.MODEL_SPECS["qwen_base"]["max_output_tokens"], 8192)
        self.assertEqual(models.MODEL_SPECS["qwen_max"]["model"], "qwencloud/qwen3.8-max")

    def test_shared_builder_uses_each_models_supported_output_budget(self):
        with patch.object(models, "build_model") as build, \
                patch.object(models.qwen_cloud, "build_model") as cloud, \
                patch.object(models, "local_reservation", return_value=None), \
                patch.object(models, "current_local_binding", return_value={"base_url": models.QWEN_BASE_URL}), \
                patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": models.QWEN_BASE_URL}):
            for name, spec in models.MODEL_SPECS.items():
                if spec.get("retired") or spec.get("requires_scientific_amendment") or spec.get("service_model_id") in {
                        models.RETRAINED_SERVICE_MODEL_ID, models.PROVIDER352_SERVICE_MODEL_ID,
                        models.SECURITY500_SERVICE_MODEL_ID, models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID}:
                    continue
                models.build_company_model(name)
                if spec.get("provider") == "qwencloud":
                    cloud.assert_called_with(spec["api_model"], 8192, 1)
                else:
                    build.assert_called_with(spec["model"], 1, 8192)
        config = hosted_model_config(models.LAST8_MODEL)
        self.assertTrue(config["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config["extra_body"]["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_stale_local_client_url_is_rejected_before_any_model_build(self):
        with patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": "http://127.0.0.1:9999/v1"}), \
                patch.object(models, "local_reservation", return_value=None), \
                patch.object(models, "current_local_binding", return_value={"base_url": models.QWEN_BASE_URL}), \
                patch.object(models, "build_model") as build:
            with self.assertRaisesRegex(ValueError, "reviewed Mac endpoint"):
                models.build_company_model("sp_lense_last8")
            build.assert_not_called()

    def test_reservation_keeps_local_choices_visible_and_releases_live(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "reservation.json"
            credentials = Path(directory) / ".env"
            write_credentials(credentials)
            with patch.dict(models.os.environ, {"RESEARCH4_QWEN_RESERVATION_FILE": str(marker), **QWEN_ENV}), \
                    patch.object(models, "current_local_binding", return_value={}):
                marker.write_text('{"active":true,"purpose":"paired evaluation"}')
                held = models.model_choices(credentials)
                self.assertEqual(len(held), 17)
                self.assertEqual({c["id"] for c in held if c["available"]},
                                 {"terra", "deepseek", "qwen_max"})
                for choice in [c for c in held if c["id"] in CURRENT_LOCAL_MODELS]:
                    self.assertIn("Temporarily reserved", choice["status"])
                with patch.object(models.urllib.request, "urlopen") as health, patch.object(models, "build_model") as build:
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        models.assert_local_binding()
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        models.build_company_model("sp_lense_last8")
                    health.assert_not_called()
                    build.assert_not_called()
                marker.write_text('{"active":false}')
                released = models.model_choices(credentials)
                self.assertEqual(sum(c["available"] for c in released), 7)

    def test_cloud_routing_never_uses_the_reserved_mac_or_its_stale_url(self):
        with patch.object(models, "local_reservation", return_value={"active": True}), \
                patch.object(models.qwen_cloud, "build_model", return_value="cloud client") as cloud, \
                patch.object(models, "build_model") as local, \
                patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": "http://127.0.0.1:9999/v1"}):
            self.assertEqual(models.build_company_model("qwen_max"), "cloud client")
            cloud.assert_called_once_with("qwen3.8-max", 8192, 1)
            local.assert_not_called()
            with self.assertRaisesRegex(ValueError, "reserved"):
                models.build_company_model("sp_lense_last8")

    def test_provider_key_lookup_is_explicit_and_cyber_cannot_bypass_selection(self):
        for key, spec in models.MODEL_SPECS.items():
            expected = None if spec.get("local_qwen") else spec["key_name"]
            self.assertEqual(models.required_api_key(spec), expected, key)
        self.assertIsNone(models.required_api_key(models.MODEL_SPECS["qwen_base"]))
        self.assertEqual(models.required_api_key(models.MODEL_SPECS["qwen_max"]), "QWEN_API_KEY")
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_sha256"], models.LAST8_SHA)
        for spec in models.MODEL_SPECS.values():
            if spec.get("provider") == "qwencloud":
                self.assertEqual(spec["max_connections"], 1)
                self.assertEqual(spec["concurrency_scope"], "account")
            else:
                self.assertEqual(spec.get("max_connections", 1), 1)
        for selection in (["gpt_cyber"], ["qwen_pro_max"], ["qwen_max", "qwen_max"], ["sp_lense"], ["sp_lense_neutral"]):
            with self.assertRaises(ValueError):
                models.selected_models(selection)

    def test_unreadable_reservation_only_holds_local_models(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "reservation.json"
            marker.write_text("partial json")
            with patch.dict(models.os.environ, {"RESEARCH4_QWEN_RESERVATION_FILE": str(marker)}):
                self.assertIsNotNone(models.local_reservation())


class Provider352PreparationTests(TestCase):
    def test_distinct_future_conditions_keep_unknown_final_bindings_and_old_identities(self):
        self.assertEqual(models.PROVIDER352_SERVICE_MODEL_ID, "sp-lense-adaptor-last-8-layers-provider352")
        self.assertNotEqual(models.PROVIDER352_MODEL, models.RETRAINED_MODEL)
        for key, condition in zip(PROVIDER352_MODELS, ("baseline", "security_maximization")):
            spec = models.MODEL_SPECS[key]
            self.assertEqual(spec["model"], models.PROVIDER352_MODEL)
            self.assertEqual(spec["service_model_id"], models.PROVIDER352_SERVICE_MODEL_ID)
            self.assertEqual(spec["prompt_condition"], condition)
            self.assertEqual(spec["provider_readiness"], "NOT_READY")
            self.assertIs(spec["execution_enabled"], False)
            for field in ("adapter_sha256", "adapter_config_sha256", "completed_training_updates", "provider_DEV_rows",
                          "four_route_source_freeze", "four_route_qualified_readiness"):
                self.assertIsNone(spec[field], field)
            self.assertEqual(spec["required_context_tokens"], 131072)
            self.assertEqual(spec["max_output_tokens"], 8192)
            self.assertEqual(spec["max_connections"], 1)
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_sha256"],
                         "d050d018d8311aaf941e837bd93b6effc7261e3e0890b7f471c851119f87929c")
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_config_sha256"],
                         "476b26d8530c275d934f5a920923885b893be3e55c63d47f4850537d1781ea9d")
        self.assertEqual(models.MODEL_SPECS["sp_lense_retrained"]["adapter_sha256"],
                         "9a4538c3bc9bea9b9a2db206b625bbbad63f1b41ef6a48d44fef1812bf51c695")
        self.assertEqual(models.MODEL_SPECS["sp_lense_retrained"]["adapter_config_sha256"],
                         "42d92a0014893d4c527b820a8d207bbeff920877ec02883f9c482e7919a8e7cb")
        self.assertIs(models.MODEL_SPECS["qwen_base"]["adapter_applied"], False)
        profile = hosted_model_config(models.LAST8_MODEL)
        self.assertIs(profile["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(profile["extra_body"]["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_catalog_keeps_future_routes_disabled_even_with_a_ready_binding_or_reservation(self):
        with patch.object(models, "current_local_binding", return_value={"expected_health": {
                "ready": True, "model_ids": [models.PROVIDER352_SERVICE_MODEL_ID]}}), \
                patch.object(models.urllib.request, "urlopen", side_effect=AssertionError("No catalog probe")):
            for reservation in (None, {"active": True, "purpose": "training"}):
                with patch.object(models, "local_reservation", return_value=reservation):
                    choices = {row["id"]: row for row in models.model_choices()}
                for key in PROVIDER352_MODELS:
                    self.assertIs(choices[key]["available"], False)
                    self.assertIn("NOT_READY", choices[key]["status"])
                    self.assertIn("partial180 cannot qualify", choices[key]["reason"])

    def test_partial_or_placeholder_metadata_cannot_admit_selection_build_or_health_probe(self):
        with patch.object(models, "current_local_binding") as binding, \
                patch.object(models, "build_model") as build, \
                patch.object(models.qwen_cloud, "build_model") as cloud, \
                patch.object(models, "local_reservation", return_value=None), \
                patch.object(models.urllib.request, "urlopen") as health:
            for count in (None, 180, 2160):
                changes = {"completed_training_updates": count, "provider_DEV_rows": 500,
                           "adapter_sha256": "a" * 64, "adapter_config_sha256": "b" * 64,
                           "provider_readiness": "READY", "execution_enabled": True}
                with patch.dict(models.MODEL_SPECS[PROVIDER352_MODELS[0]], changes), \
                        patch.dict(models.MODEL_SPECS[PROVIDER352_MODELS[1]], changes):
                    for key in PROVIDER352_MODELS:
                        for operation in (models.selected_models, models.build_company_model, models.assert_local_binding):
                            with self.subTest(count=count, model=key, operation=operation.__name__), \
                                    self.assertRaisesRegex(ValueError, "NOT_READY"):
                                operation([key] if operation is models.selected_models else key)
            binding.assert_not_called()
            build.assert_not_called()
            cloud.assert_not_called()
            health.assert_not_called()

    def test_four_route_binding_stays_held_before_receipt_access_or_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = new_binding_fixture(Path(directory))
            for schema in (models.PROVIDER352_BINDING_SCHEMA, models.COMPARISON_BINDING_SCHEMA, None):
                binding = copy.deepcopy(original)
                binding["schema"] = schema
                binding["readiness_receipt_path"] = str(Path(directory) / "unqualified_missing_receipt.json")
                if schema != models.PROVIDER352_BINDING_SCHEMA:
                    binding["expected_health"]["model_ids"].append(models.PROVIDER352_SERVICE_MODEL_ID)
                path.write_text(json.dumps(binding))
                with patch.object(models.urllib.request, "urlopen") as health, \
                        self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "NOT_READY"):
                    models.current_local_binding(path)
                health.assert_not_called()


class Provider352FixturePortabilityTests(TestCase):
    def test_only_absent_packet_skips_while_present_missing_or_tampered_packets_fail(self):
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "read_bytes") as read:
            with self.assertRaisesRegex(SkipTest, "Local Root handoff integration"):
                provider352_consumer_fixture()
            read.assert_not_called()
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_bytes", side_effect=FileNotFoundError("incomplete local packet")):
            with self.assertRaises(FileNotFoundError):
                provider352_consumer_fixture()
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_bytes", return_value=b"tampered"):
            with self.assertRaises(AssertionError):
                provider352_consumer_fixture()


class DormantProvider352ConsumerTests(TestCase):
    def test_exact_shared_helper_and_actual_composed_contract_cover_all_four_routes(self):
        fixture = provider352_consumer_fixture()
        for route in fixture.consumer.contract.ARMS:
            self.assertIs(fixture.consumer.validate_request(route)["metadata_request_allowed"], True)
        self.assertEqual(fixture.reads.count("current-window.json"), 4)
        self.assertEqual(fixture.health.call_count, 4)
        with patch.object(models, "PROVIDER352_HANDOFF_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Exact Root"):
                models.Provider352GameConsumer(fixture.pins, core_binding=fixture.core_binding,
                                               requester_chat=fixture.consumer.requester_chat)
        self.assertNotIn("validator", models.Provider352GameConsumer.validate_request.__code__.co_varnames)

    def test_actual_http_attempt_guard_preserves_hooks_client_profile_and_retry_settings(self):
        fixture = provider352_consumer_fixture()
        dispatched, original_hooks = [], []
        async def existing_hook(request):
            original_hooks.append(request)
        async def attempt(client, request):
            for hook in client.event_hooks["request"]:
                await hook(request)
            dispatched.append(request)
        for model_id in ("qwen_base", "sp_lense_last8", "sp_lense_retrained", "sp_lense_provider352"):
            values = ({"adapter_sha256": models.PROVIDER352_TECHNICAL_PINS["adapter"],
                       "adapter_config_sha256": models.PROVIDER352_TECHNICAL_PINS["adapter_config"]}
                      if model_id == "sp_lense_provider352" else {})
            with patch.dict(models.MODEL_SPECS[model_id], values):
                model, client = provider352_fake_http_model(model_id)
                client.event_hooks["request"].append(existing_hook)
                before = (client.transport, client.cache_marker, client.retries,
                          client.event_hooks["response"], vars(model.config).copy())
                self.assertIs(fixture.consumer.install_http_guard(model, model_id), model)
                wire = provider352_fake_wire(fixture, model_id)
                raw = wire.content
                asyncio.run(attempt(client, wire))
                asyncio.run(attempt(client, wire))  # Another actual attempt, including an SDK retry.
                self.assertEqual(wire.content, raw)
                self.assertIs(client.event_hooks["request"][0], existing_hook)
                self.assertEqual((client.transport, client.cache_marker, client.retries,
                                  client.event_hooks["response"], vars(model.config)), before)
        self.assertEqual(len(dispatched), 8)
        self.assertEqual(len(original_hooks), 8)
        self.assertEqual(fixture.reads.count("current-window.json"), 8)
        self.assertEqual(fixture.health.call_count, 8)
        self.assertIsNone(models.MODEL_SPECS["sp_lense_provider352"]["adapter_sha256"])

    def test_changed_window_latches_denial_before_dispatch_and_cannot_silently_resume(self):
        fixture = provider352_consumer_fixture()
        model, client = provider352_fake_http_model("qwen_base")
        fixture.consumer.install_http_guard(model, "qwen_base")
        wire = provider352_fake_wire(fixture, "qwen_base")
        old = fixture.records["current-window.json"]
        revoked = copy.deepcopy(fixture.window)
        revoked.update(status="REVOKED", permissions={key: False for key in revoked["permissions"]})
        fixture.records["current-window.json"] = fixture.raw(revoked)
        with self.assertRaisesRegex(ValueError, "denied"):
            asyncio.run(client.event_hooks["request"][0](wire))
        fixture.health.assert_not_called()
        fixture.records["current-window.json"] = old
        with self.assertRaisesRegex(ValueError, "stopped"):
            asyncio.run(client.event_hooks["request"][0](wire))
        self.assertEqual(fixture.reads.count("current-window.json"), 1)

    def test_real_httpx2_mock_transport_is_guarded_and_denial_is_operational_incomplete(self):
        try:
            import httpx2
        except ImportError:
            self.skipTest("SDK httpx2 is not installed")
        from src.hosted_inspect_runner import run_episode, outcome_label
        from tests.test_hosted_inspect_runner import FakeSimulator

        fixture = provider352_consumer_fixture()
        dispatched, existing = [], []
        async def first_hook(request):
            existing.append(request)
        async def handler(request):
            dispatched.append(request)
            return httpx2.Response(200, json={"synthetic": True})
        async def exercise():
            transport = httpx2.MockTransport(handler)
            async with httpx2.AsyncClient(transport=transport, event_hooks={"request": [first_hook]}) as client:
                model, _ = provider352_fake_http_model("qwen_base")
                model.api.client._client = client
                wire = provider352_fake_wire(fixture, "qwen_base")
                async def generate(*args, **kwargs):
                    try:
                        return await client.post(wire.url, content=wire.content)
                    except Exception:
                        raise RuntimeError("Synthetic SDK error wrapper") from None
                model.generate = generate
                fixture.consumer.install_http_guard(model, "qwen_base")
                await model.generate([])
                self.assertIs(client._transport, transport)
                self.assertIs(client.event_hooks["request"][0], first_hook)
                fixture.records["current-window.json"] = b'{"status":"REVOKED"}'
                sim = FakeSimulator()
                async def request(prompt):
                    return await model.generate([])
                result = await run_episode(sim, request, max_decisions=1, timeout_seconds=2)
                self.assertEqual(outcome_label(result), "incomplete")
                self.assertEqual(sim.received, [])
                self.assertEqual(sim.decisions, 0)
        asyncio.run(exercise())
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(len(existing), 1)

    def test_generate_cache_entry_and_suppressed_hook_error_cannot_return_a_completion(self):
        from src.hosted_inspect_runner import IncompleteResponse

        fixture = provider352_consumer_fixture()
        model, client = provider352_fake_http_model("qwen_base")
        wire = provider352_fake_wire(fixture, "qwen_base")
        calls = []
        async def suppressing_sdk(*args, **kwargs):
            calls.append("delegate")
            fixture.records["current-window.json"] = b'{"status":"REVOKED"}'
            try:
                await client.event_hooks["request"][-1](wire)
            except ValueError:
                return "normal completion after suppressed hook error"
        model.generate = suppressing_sdk
        fixture.consumer.install_http_guard(model, "qwen_base")
        with self.assertRaises(IncompleteResponse):
            asyncio.run(model.generate([]))
        self.assertEqual(calls, ["delegate"])

        cached = provider352_consumer_fixture()
        cached_model, _ = provider352_fake_http_model("qwen_base")
        cache_calls = []
        async def cache_hit(*args, **kwargs):
            cache_calls.append("cache")
            return "cached completion without HTTP"
        cached_model.generate = cache_hit
        cached.consumer.install_http_guard(cached_model, "qwen_base")
        self.assertEqual(asyncio.run(cached_model.generate([])), "cached completion without HTTP")
        cached.records["current-window.json"] = b'{"status":"REVOKED"}'
        with self.assertRaises(IncompleteResponse):
            asyncio.run(cached_model.generate([]))
        self.assertEqual(cache_calls, ["cache"])

    def test_wire_route_profile_bounds_and_unbound_provider_are_rejected(self):
        for mutation in (lambda body: body.update(model=models.RETRAINED_SERVICE_MODEL_ID),
                         lambda body: body.update(temperature=0.2),
                         lambda body: body.update(max_tokens=4096),
                         lambda body: body.update(chat_template_kwargs={"enable_thinking": False}),
                         lambda body: body.update(stream=False)):
            fixture = provider352_consumer_fixture()
            model, client = provider352_fake_http_model("qwen_base")
            fixture.consumer.install_http_guard(model, "qwen_base")
            wire = provider352_fake_wire(fixture, "qwen_base")
            body = json.loads(wire.content)
            mutation(body)
            wire.content = json.dumps(body).encode()
            with self.assertRaisesRegex(ValueError, "denied"):
                asyncio.run(client.event_hooks["request"][0](wire))
            fixture.health.assert_not_called()
        for model_id, wrong_url in (("qwen_base", True), ("sp_lense_provider352", False)):
            fixture = provider352_consumer_fixture()
            model, client = provider352_fake_http_model(model_id)
            fixture.consumer.install_http_guard(model, model_id)
            wire = provider352_fake_wire(fixture, model_id)
            if wrong_url:
                wire.url = "http://127.0.0.1:9999/v1/chat/completions"
            with self.assertRaisesRegex(ValueError, "denied"):
                asyncio.run(client.event_hooks["request"][0](wire))
            fixture.health.assert_not_called()

    def test_expiry_fresh_health_review_and_held_metadata_cannot_pass(self):
        for failure in ("expiry", "health", "review", "held", "missing"):
            fixture = provider352_consumer_fixture()
            if failure == "expiry":
                fixture.clock.return_value += 7200
            elif failure == "health":
                fixture.health.side_effect = lambda core: {**core["expected_health"], "pid": 1}
            elif failure == "review":
                role = fixture.core["provider_artifacts"]["completion_auditor_review"]["path"]
                fixture.records[role] = b'{"status":"focused-only"}'
            elif failure == "held":
                index = {**fixture.index, "ready": False, "status": "HELD_NOT_READY"}
                fixture.records["index.json"] = fixture.raw(index)
            else:
                del fixture.records["current-window.json"]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                fixture.consumer.validate_request(models.PROVIDER352_SERVICE_MODEL_ID)
        root_requester = provider352_consumer_fixture(requester_chat="Root")
        with self.assertRaisesRegex(ValueError, "denied"):
            root_requester.consumer.validate_request(models.PROVIDER352_SERVICE_MODEL_ID)

    def test_trusted_scope_is_copied_and_production_build_remains_dormant(self):
        fixture = provider352_consumer_fixture()
        fixture.pins["expected_nonce"] = "f" * 32
        self.assertEqual(fixture.consumer.pins["expected_nonce"], "e" * 32)
        parent_sha = fixture.consumer.pins["expected_game_plan_sha256"]
        private = fixture.consumer.private_child_context(parent_sha)
        self.assertEqual(json.loads(private)["pins"]["expected_game_plan_sha256"], parent_sha)
        self.assertEqual(json.loads(private)["requester_chat"], fixture.consumer.requester_chat)
        with self.assertRaisesRegex(ValueError, "Whole parent"):
            fixture.consumer.require_plan("6" * 64)
        with patch.dict(models.os.environ, {models.PROVIDER352_CONTEXT_ENV: private}), \
                patch.object(models, "current_local_binding") as binding, \
                patch.object(models, "build_model") as build, \
                patch.object(models.urllib.request, "urlopen") as health:
            for model_id in ("qwen_base", "sp_lense_provider352", "sp_lense_provider352_neutral"):
                with self.assertRaisesRegex(ValueError, "NOT_READY"):
                    models.build_company_model(model_id)
            binding.assert_not_called()
            build.assert_not_called()
            health.assert_not_called()


class Provider352ActivationTests(TestCase):
    def setUp(self):
        provider352_isolate_state(self)

    def test_planning_only_core_updates_shared_registry_without_grant_health_or_build(self):
        fixture = provider352_consumer_fixture()
        from src import company_world_evaluation as evaluation
        from src import infrastructure_preservation_comparison as comparison
        old = {name: copy.deepcopy(models.MODEL_SPECS[name]) for name in CURRENT_LOCAL_MODELS +
               ("sp_lense_retrained_neutral", "sp_lense_retrained")}
        registry = models.MODEL_SPECS
        with patch.object(models, "_provider352_fresh_health", side_effect=AssertionError("No planning health")), \
                patch.object(models, "build_model") as build:
            core = models.initialize_provider352_core(fixture.core_binding, read_bytes=fixture.read_metadata)
            self.assertEqual(fixture.reads, ["core.json"])
            self.assertEqual(models.selected_models(list(PROVIDER352_MODELS)), list(PROVIDER352_MODELS))
            self.assertIs(evaluation.MODEL_SPECS, registry)
            self.assertIs(comparison.MODEL_SPECS, registry)
            for name in PROVIDER352_MODELS:
                spec = models.MODEL_SPECS[name]
                self.assertEqual(spec["adapter_sha256"], models.PROVIDER352_TECHNICAL_PINS["adapter"])
                self.assertEqual(spec["provider_readiness"], "PLANNING_ONLY")
                self.assertNotIn("execution_enabled", spec)
                self.assertFalse({"grant_nonce", "permissions", "expires_utc"} & set(spec))
                condition = "FT only" if spec["prompt_condition"] == "baseline" else "FT + security instruction"
                self.assertEqual(spec["label"], "SP Lens -2026-11-14 23:39 ABT · " + condition)
            self.assertEqual(models.current_local_binding(required_models=list(PROVIDER352_MODELS)), core)
            for operation in (models.build_company_model, models.assert_local_binding):
                with self.assertRaisesRegex(ValueError, "NOT_READY"):
                    operation(PROVIDER352_MODELS[0])
            choices = {row["id"]: row for row in models.model_choices()}
            self.assertIs(choices[PROVIDER352_MODELS[0]]["available"], False)
            build.assert_not_called()
        self.assertEqual({name: models.MODEL_SPECS[name] for name in old}, old)
        fixture.health.assert_not_called()

    def test_wrong_technical_historical_partial_or_old_chain_core_never_initializes(self):
        fixture = provider352_consumer_fixture()
        mutations = [lambda core: core.update(ready=False),
            lambda core: (core["per_model"][models.PROVIDER352_SERVICE_MODEL_ID].update(
                              adapter_sha256="1" * 64, adapter_config_sha256="2" * 64),
                          core["provider_artifacts"]["adapter"].update(sha256="1" * 64),
                          core["provider_artifacts"]["adapter_config"].update(sha256="2" * 64)),
            lambda core: core["provider_artifacts"]["completion_audit"].update(sha256="a" * 64),
            lambda core: core["provider_artifacts"].update(readiness=None),
            lambda core: core["provider_artifacts"]["completion_auditor_review"].update(
                sha256="8185809f6a5df00c84fe88bad7134e389f3b11b3531440d3c7d534a7380fb5aa")]
        for mutate in mutations:
            core = copy.deepcopy(fixture.core)
            mutate(core)
            raw = fixture.raw(core)
            reference = {"path": "core.json", "sha256": hashlib.sha256(raw).hexdigest()}
            with self.assertRaises(ValueError):
                models.initialize_provider352_core(reference, read_bytes=lambda path: raw)
            self.assertIsNone(models.provider352_planning_core_ref())
            self.assertIsNone(models.MODEL_SPECS[PROVIDER352_MODELS[0]]["adapter_sha256"])

    def test_bad_display_name_rejects_before_either_provider_spec_or_core_ref_changes(self):
        fixture = provider352_consumer_fixture()
        before = {name: copy.deepcopy(models.MODEL_SPECS[name]) for name in PROVIDER352_MODELS}
        for label in (None, 17, ""):
            core = copy.deepcopy(fixture.core)
            if label is None:
                core["model_display_names"].pop(models.PROVIDER352_SERVICE_MODEL_ID)
            else:
                core["model_display_names"][models.PROVIDER352_SERVICE_MODEL_ID] = label
            raw = fixture.raw(core)
            reference = {"path": "core.json", "sha256": hashlib.sha256(raw).hexdigest()}
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "display name"):
                models.initialize_provider352_core(reference, read_bytes=lambda path: raw)
            self.assertEqual({name: models.MODEL_SPECS[name] for name in PROVIDER352_MODELS}, before)
            self.assertIsNone(models.provider352_planning_core_ref())

    def test_private_child_context_builds_guarded_local_clients_and_preserves_hosted_transport(self):
        fixture = provider352_consumer_fixture()
        private = fixture.consumer.private_child_context(fixture.pins["expected_game_plan_sha256"])
        clients = []
        def build(model_name, connections, output):
            name = next((name for name, spec in models.MODEL_SPECS.items() if spec["model"] == model_name), None)
            if name and models.MODEL_SPECS[name].get("local_qwen"):
                model, client = provider352_fake_http_model(name)
                clients.append(client)
                return model
            return "existing hosted transport"
        with patch.dict(models.os.environ, {models.PROVIDER352_CONTEXT_ENV: private,
                                           "RESEARCH4_QWEN_BASE_URL": fixture.core["base_url"]}), \
                patch.object(models, "_bounded_metadata_bytes", side_effect=fixture.read_metadata), \
                patch.object(models, "_provider352_fresh_health", side_effect=fixture.health), \
                patch.object(models.time, "time", side_effect=fixture.clock), \
                patch.object(models, "local_reservation", return_value=None), \
                patch.object(models, "build_model", side_effect=build) as constructor:
            for name in ("qwen_base", "sp_lense_last8", "sp_lense_retrained", "sp_lense_provider352"):
                models.build_company_model(name)
            self.assertEqual([len(client.event_hooks["request"]) for client in clients], [1, 1, 1, 1])
            for name in ("terra", "deepseek"):
                self.assertEqual(models.build_company_model(name), "existing hosted transport")
            self.assertEqual(models.assert_local_binding("last8"), fixture.core["expected_health"])
            self.assertEqual(constructor.call_count, 6)
            changed = json.loads(private)
            changed["pins"]["expected_nonce"] = "f" * 32
            with patch.dict(models.os.environ, {models.PROVIDER352_CONTEXT_ENV: json.dumps(changed)}):
                with self.assertRaisesRegex(ValueError, "scope changed"):
                    models.build_company_model("qwen_base")
            self.assertEqual(constructor.call_count, 6)

    def test_core_and_registry_cannot_rebind_and_index_cannot_replace_independent_core(self):
        fixture = provider352_consumer_fixture()
        models.initialize_provider352_core(fixture.core_binding, read_bytes=fixture.read_metadata)
        with self.assertRaisesRegex(ValueError, "immutable"):
            models.initialize_provider352_core({**fixture.core_binding, "path": "replacement.json"},
                                              read_bytes=fixture.read_metadata)
        with patch.dict(models.MODEL_SPECS[PROVIDER352_MODELS[0]], adapter_sha256=models.LAST8_SHA):
            with self.assertRaisesRegex(ValueError, "NOT_READY"):
                models.selected_models([PROVIDER352_MODELS[0]])
        index = copy.deepcopy(fixture.index)
        index["core_binding"]["path"] = "replacement.json"
        raw = fixture.raw(index)
        fixture.records["index.json"] = raw
        pins = {**fixture.pins, "expected_index_sha256": hashlib.sha256(raw).hexdigest()}
        consumer = models.Provider352GameConsumer(pins, core_binding=fixture.core_binding,
            requester_chat=fixture.consumer.requester_chat, read_bytes=fixture.read_metadata,
            read_health=fixture.health, clock=fixture.clock)
        with self.assertRaisesRegex(ValueError, "denied"):
            consumer.validate_request(models.PROVIDER352_SERVICE_MODEL_ID)
        fixture.health.assert_not_called()


class LocalBaseRoutingEvidenceTests(TestCase):
    def test_all_four_current_local_choices_share_the_new_paired_base(self):
        for key in CURRENT_LOCAL_MODELS:
            spec = models.MODEL_SPECS[key]
            self.assertTrue(spec["local_qwen"])
            self.assertEqual(spec["backend"], "local_mlx")
            self.assertEqual(spec["base_quantization"], "4bit affine/group64; not NF4")
            self.assertEqual(spec["base_revision"], models.LOCAL_BASE_REVISION)
            self.assertEqual(spec["base_manifest_sha256"], models.LOCAL_BASE_MANIFEST_SHA256)
            self.assertEqual(spec["max_output_tokens"], 8192)
            self.assertEqual(spec["local_binding_group"], "last8")
            self.assertIn("fresh health readback required", spec["configuration_evidence"])
        self.assertFalse(models.MODEL_SPECS["qwen_base"]["adapter_applied"])
        self.assertIsNone(models.MODEL_SPECS["qwen_base"]["adapter_sha256"])
        self.assertTrue(models.MODEL_SPECS["sp_lense_last8"]["adapter_applied"])
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_sha256"], models.LAST8_SHA)
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_config_sha256"], models.LAST8_CONFIG_SHA)
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_rank"], 16)
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_blocks"], list(range(56, 64)))
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["completed_training_updates"], 2160)
        self.assertEqual(models.MODEL_SPECS["sp_lense"]["adapter_sha256"], models.LAST_SHA)
        for key in ("qwen_max", "terra", "deepseek"):
            self.assertEqual(models.MODEL_SPECS[key]["precision"], "unknown")

    def test_reserved_local_base_has_no_hosted_fallback_or_endpoint_probe(self):
        with patch.object(models, "local_reservation", return_value={"active": True, "purpose": "training"}), \
                patch.object(models, "build_model") as local, \
                patch.object(models.qwen_cloud, "build_model") as cloud, \
                patch.object(models.urllib.request, "urlopen") as health:
            for key in CURRENT_LOCAL_MODELS:
                with self.assertRaisesRegex(ValueError, "reserved"):
                    models.build_company_model(key)
            local.assert_not_called()
            cloud.assert_not_called()
            health.assert_not_called()

    def test_wrong_retained_base_identity_is_rejected_without_a_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "readiness.json"
            receipt.write_text(json.dumps({"base": {
                "revision": models.LOCAL_BASE_REVISION,
                "manifest_sha256": models.LOCAL_BASE_MANIFEST_SHA256},
                "LAST": {"adapter_sha256": models.LAST_SHA},
                "numerical_policy": {"platform": "Mac native MLX",
                                     "base_quantization": "4bit affine/group64; not NF4"}}))
            original = {"adapter_sha256": models.LAST_SHA,
                "base_url": models.QWEN_BASE_URL, "health_url": "http://127.0.0.1:8769/health",
                "readiness_receipt_path": str(receipt),
                "readiness_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "expected_health": {"adapter_sha256": models.LAST_SHA, "ready": True,
                    "base_revision": models.LOCAL_BASE_REVISION,
                    "base_manifest_sha256": models.LOCAL_BASE_MANIFEST_SHA256,
                    "model_ids": ["qwen3.8-27b-base", "sp-lense-adaptor-last-layer"],
                    "max_concurrency": 1, "resident_base_count": 1}}
            binding = root / "binding.json"
            binding.write_text(json.dumps(original))
            with patch.object(models.urllib.request, "urlopen") as health:
                self.assertEqual(models.local_binding(binding), original)
                for key in ("base_revision", "base_manifest_sha256"):
                    changed = {**original, "expected_health": {**original["expected_health"], key: "wrong"}}
                    binding.write_text(json.dumps(changed))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reviewed 4-bit base"):
                        models.local_binding(binding)
                health.assert_not_called()


class CurrentLast8BindingTests(TestCase):
    def test_missing_new_handoff_cannot_reuse_historical_last_or_claim_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / ".env"
            write_credentials(credentials)
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(root / "missing.json"), **QWEN_ENV}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models, "local_binding", side_effect=AssertionError("Historical fallback forbidden")), \
                    patch.object(models, "build_model") as build, \
                    patch.object(models.qwen_cloud, "build_model") as cloud, \
                    patch.object(models.urllib.request, "urlopen") as health:
                choices = models.model_choices(credentials)
                self.assertEqual({row["id"] for row in choices if row["available"]}, {"terra", "deepseek", "qwen_max"})
                self.assertFalse({"sp_lense", "sp_lense_neutral"} & {row["id"] for row in choices})
                for key in CURRENT_LOCAL_MODELS:
                    choice = next(row for row in choices if row["id"] == key)
                    self.assertIn("binding required", choice["status"])
                    with self.assertRaises(FileNotFoundError):
                        models.build_company_model(key)
                build.assert_not_called()
                cloud.assert_not_called()
                health.assert_not_called()

    def test_new_binding_is_configuration_only_and_supports_its_explicit_port(self):
        with tempfile.TemporaryDirectory() as directory:
            path, binding = new_binding_fixture(Path(directory), port=8871)
            with patch.object(models.urllib.request, "urlopen") as health:
                self.assertEqual(models.current_local_binding(path), binding)
                health.assert_not_called()
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(path),
                                               "RESEARCH4_QWEN_BASE_URL": binding["base_url"]}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models, "build_model") as build, \
                    patch.object(models.urllib.request, "urlopen") as health:
                models.build_company_model("sp_lense_last8")
                build.assert_called_once_with(models.LAST8_MODEL, 1, 8192)
                health.assert_not_called()

    def test_stale_adapter_config_routes_caps_or_receipt_fail_before_any_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = new_binding_fixture(root)
            mutations = [
                ("top_old_adapter", lambda x: x.update(adapter_sha256=models.LAST_SHA)),
                ("old_adapter", lambda x: x["expected_health"].update(adapter_sha256=models.LAST_SHA)),
                ("wrong_config", lambda x: x["expected_health"].update(adapter_config_sha256="wrong")),
                ("wrong_base", lambda x: x["expected_health"].update(base_revision="wrong")),
                ("wrong_manifest", lambda x: x["expected_health"].update(base_manifest_sha256="wrong")),
                ("old_route", lambda x: x["expected_health"].update(model_ids=["qwen3.8-27b-base", "sp-lense-adaptor-last-layer"])),
                ("extra_route", lambda x: x["expected_health"]["model_ids"].append("unreviewed")),
                ("two_bases", lambda x: x["expected_health"].update(resident_base_count=2)),
                ("concurrent", lambda x: x["expected_health"].update(max_concurrency=2)),
                ("small_output", lambda x: x["expected_health"]["limits"].update(max_output_tokens=4096)),
                ("stale_32k_context", lambda x: x["expected_health"]["limits"].update(
                    context_total_tokens=32768, max_context_tokens_including_output=32768)),
                ("conflicting_context", lambda x: x["expected_health"]["limits"].update(context_total_tokens=16384)),
                ("not_ready", lambda x: x["expected_health"].update(ready=False)),
                ("remote", lambda x: x.update(base_url="http://example.com:8871/v1")),
                ("implicit_port", lambda x: x.update(base_url="http://localhost/v1")),
                ("different_health_port", lambda x: x.update(health_url="http://127.0.0.1:8872/health")),
                ("query", lambda x: x.update(base_url="http://127.0.0.1:8871/v1?route=other")),
                ("wrong_receipt", lambda x: x.update(readiness_receipt_sha256="wrong")),
            ]
            with patch.object(models.urllib.request, "urlopen") as health:
                for name, mutate in mutations:
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    path.write_text(json.dumps(changed))
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        models.current_local_binding(path)
                health.assert_not_called()

    def test_retired_ids_cannot_build_or_be_relabelled_as_new_adapter(self):
        with patch.object(models, "build_model") as build, \
                patch.object(models, "current_local_binding") as binding, \
                patch.object(models.qwen_cloud, "build_model") as cloud, \
                patch.object(models.urllib.request, "urlopen") as health:
            for key in ("sp_lense", "sp_lense_neutral"):
                self.assertEqual(models.MODEL_SPECS[key]["adapter_sha256"], models.LAST_SHA)
                self.assertEqual(models.MODEL_SPECS[key]["model"], models.LAST_MODEL)
                with self.assertRaisesRegex(ValueError, "retired"):
                    models.build_company_model(key)
            build.assert_not_called()
            binding.assert_not_called()
            cloud.assert_not_called()
            health.assert_not_called()

    def test_group_fresh_readback_checks_new_identity_and_keeps_stale_url_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            _, binding = new_binding_fixture(Path(directory))
            with patch.object(models, "current_local_binding", return_value=binding), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": binding["base_url"]}), \
                    patch.object(models.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(binding["expected_health"]).encode())) as health:
                self.assertEqual(models.assert_local_binding("last8"), binding["expected_health"])
                health.assert_called_once_with(binding["health_url"], timeout=5)
            stale = {**binding["expected_health"], "adapter_sha256": models.LAST_SHA}
            with patch.object(models, "current_local_binding", return_value=binding), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": binding["base_url"]}), \
                    patch.object(models.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(stale).encode())):
                with self.assertRaisesRegex(ValueError, "BASE/last8"):
                    models.assert_local_binding("sp_lense_last8")
            with patch.object(models, "current_local_binding", return_value=binding), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": "http://127.0.0.1:9999/v1"}), \
                    patch.object(models.urllib.request, "urlopen") as health:
                with self.assertRaisesRegex(ValueError, "reviewed Mac endpoint"):
                    models.assert_local_binding("qwen_base")
                health.assert_not_called()
