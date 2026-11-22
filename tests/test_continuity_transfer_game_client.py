"""CPU-only held264 integration; no model, Mac, real HTTP or world draw."""
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import company_models as models
from src import infrastructure_preservation_comparison as runner
from src.hosted_inspect_runner import hosted_model_config
from tests.support.local_artifacts import require_local_files


FT = "sp_lense_transfer_neutral"
BASE = "qwen_base_neutral"


def core_fixture():
    """Synthetic positive core metadata, never a published readiness record."""
    require_local_files(models.CONTINUITY_TRANSFER_PUBLIC_PATH, models.CONTINUITY_TRANSFER_HANDOFF_PATH,
                        models.CONTINUITY_TRANSFER_HANDOFF_PATH.with_name("SERVICE_CORE_HELD_TEMPLATE_v1.json"))
    h = models.continuity_transfer_handoff_contract()
    core = json.loads(models.CONTINUITY_TRANSFER_HANDOFF_PATH.with_name(
        "SERVICE_CORE_HELD_TEMPLATE_v1.json").read_bytes())
    core.update(status="QUALIFIED_TWO_ROUTE_BINDING", ready=True,
                created_utc="1970-01-01T00:01:00+00:00",
                base_url="http://127.0.0.1:8769/v1", health_url="http://127.0.0.1:8769/health")
    for role in ("readiness_receipt", "two_route_qualification", "service_source_freeze"):
        core[role] = {"path": ".local/synthetic264/" + role + ".json", "sha256": "a" * 64}
    core["expected_health"] = {
        "pid": 12345, "process_created_at": 100.0,
        "source_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "ready": True, "last_post_request_check": "passed", "model_ids": list(h.ARMS),
        "per_model": core["per_model"], "base_manifest_sha256": h.BASE_SHA,
        "base_load_count": 1, "adapter_family_count": 1, "max_concurrency": 1,
        "limits": {"context_total_tokens": 131072, "default_output_tokens": 256,
                   "max_output_tokens": 8192, "request_seconds": 900,
                   "available_memory_floor_bytes": 4294967296}}
    raw = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    descriptor = {"core_binding": {"path": ".local/synthetic264/core.json",
                                   "sha256": hashlib.sha256(raw).hexdigest()},
                  "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}
    return h, core, raw, descriptor


def pins_fixture():
    return {"index_path": ".local/synthetic264/index.json",
            "expected_index_sha256": "a" * 64, "expected_window_sha256": "b" * 64,
            "expected_nonce": "c" * 32, "expected_game_plan_sha256": "d" * 64,
            "expected_direct_authority_sha256": "114bf5e1c42e059f048e5a2033aa9042942c565e4a115a3fe757173a4f10fedb"}


def completion_evidence(h, core, request_id, messages_sha256, route, ordinal=1):
    """Synthetic same-request proof; no native/tokenizer execution is claimed."""
    health = core["expected_health"]
    native = hashlib.sha256((request_id + "native").encode()).hexdigest()[:32]
    value = dict.fromkeys(h.COMPLETION_FIELDS)
    value.update(schema=h.COMPLETION_SCHEMA, request_id=request_id, native_ticket_id=native,
        model_id_requested=route, model_id_serviced=route, model_id_restored=h.BASE,
        adapter_sha256_during_generation=h.FINAL_ADAPTER_SHA if route == h.FT else None,
        base_manifest_sha256=h.BASE_SHA, service_pid=health["pid"],
        service_process_created_at=health["process_created_at"],
        service_source_sha256=health["source_sha256"], manifest_sha256=health["manifest_sha256"],
        messages_sha256=messages_sha256, seed=20261108,
        seed_reset_native_path="mlx_lm.server.ResponseGenerator._serve_single:mx.random.seed(args.seed)",
        native_server_sha256=h.NATIVE_SERVER_SHA, max_tokens=8192, prompt_tokens=20,
        completion_tokens=2, actual_generated_token_ids=[4, h.CANONICAL_EOS],
        eos_token_ids=list(h.EOS_IDS), canonical_eos_token_id=h.CANONICAL_EOS,
        tokenizer_metadata_sha256=h.TOKENIZER_METADATA["sha256"], vocab_size=h.VOCAB_SIZE,
        finish_reason="stop", EOS_detected=True, KV_cache_empty=True,
        BASE_restored_and_verified=True, adapter_wrappers_after=0, completed_epoch=100.0,
        completion_ack={"request_id": native, "worker_started": True, "worker_finished": True,
                        "backend_barrier_done": True, "post_checks_passed": True, "native_errors": [],
                        "worker_thread_id": 100, "owned_stream_object_id": 200, "context_attached": True,
                        "stop_requested": True, "helper_stop_sent": True},
        dispatch_record={"phase": "GAME_WINDOW", "request_id": request_id, "window_sha256": "d" * 64,
                         "game_dispatch_ordinal": ordinal, "max_game_requests": 1536})
    return value


def transport_fixture(model_id=FT, *, global_expiry=10000):
    """Bridge fixture: mock admission metadata; run the exact real post-join helper."""
    _, core, _, _ = core_fixture()
    h = models.continuity_transfer_handoff_contract()
    empty = b"{}"
    empty_sha = hashlib.sha256(empty).hexdigest()
    for role in ("readiness_receipt", "two_route_qualification"):
        core[role]["sha256"] = empty_sha
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    reference = lambda path, raw: {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    core_raw = encode(core)
    descriptor = {"core_binding": reference(".local/synthetic264/core.json", core_raw),
                  "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}
    public = json.loads(models.CONTINUITY_TRANSFER_PUBLIC_PATH.read_bytes())
    window = json.loads((models.PROVIDER352_ROOT / public["held_templates"]["window"]["path"]).read_bytes())
    window.update(status="GRANTED_TO_GAME", issued_by_root=True, owner_chat=h.GAME_CHAT,
        grant_nonce="c" * 32, issued_utc="1970-01-01T00:00:00+00:00",
        expires_utc=datetime.fromtimestamp(global_expiry, timezone.utc).isoformat(),
        core_binding=descriptor["core_binding"], game_plan_sha256="d" * 64,
        max_game_requests=1536, prior_window_drain_receipt={"path": ".local/synthetic264/drain.json", "sha256": "a" * 64})
    window["permissions"] = {key: key in {"game_collection", "game_model_requests"} for key in h.PERMISSIONS}
    window_raw = encode(window)
    index = {"schema": h.INDEX_SCHEMA, "status": "READY_FOR_GAME_BOUND_WINDOW", "ready": True,
             "core_binding": descriptor["core_binding"],
             "request_window": reference(".local/synthetic264/window.json", window_raw), "scope": "synthetic"}
    index_raw = encode(index)
    pins = pins_fixture()
    pins.update(expected_index_sha256=hashlib.sha256(index_raw).hexdigest(),
                expected_window_sha256=hashlib.sha256(window_raw).hexdigest())
    records = {descriptor["core_binding"]["path"]: core_raw,
        index["request_window"]["path"]: window_raw, pins["index_path"]: index_raw,
        core["readiness_receipt"]["path"]: empty, core["two_route_qualification"]["path"]: empty}
    # Actual saved JSON-only audit/review are immutable known metadata, never weights.
    for role in ("completion_audit", "completion_auditor_review"):
        ref = core["artifacts"][role]
        records[ref["path"]] = (models.PROVIDER352_ROOT / ref["path"]).read_bytes()
    health = copy.deepcopy(core["expected_health"])
    health.update(updated_epoch=100.0, pending_http_requests=0,
                  completion_gate={"active_request_id": None, "finishing": False, "unavailable_reason": None})
    reader = lambda path: records[Path(path).relative_to(models.PROVIDER352_ROOT).as_posix()]
    consumer = models.ContinuityTransferGameConsumer(pins, core_binding=descriptor,
        requester_chat=h.GAME_CHAT, read_bytes=reader, read_health=lambda core, *, timeout: copy.deepcopy(health),
        clock=lambda: 100.0, monotonic=lambda: 100.0)
    decision = {"metadata_request_allowed": True, "owner_chat": h.GAME_CHAT,
                "model_id": models.MODEL_SPECS[model_id]["service_model_id"], "grant_nonce": window["grant_nonce"]}
    # Root's independent pure-helper tests cover admission qualification. These
    # bridge tests isolate actual wire construction and the real completion join.
    consumer.contract.validate_per_request_metadata = Mock(return_value=decision)
    consumer.contract._validate_bound_metadata = Mock(return_value=decision)
    consumer._child_plan_verified = True
    consumer._validate_game_source_scope = Mock(return_value={"models": [model_id]})
    consumer.bind_case_deadline("1970-01-01T00:01:00+00:00", "1970-01-01T04:01:00+00:00")
    return SimpleNamespace(consumer=consumer, core=core, health=health, model_id=model_id,
                           window=window, pins=pins, records=records)


def fake_transport_model(fixture, client, result_stop="stop", after_http=None):
    spec = models.MODEL_SPECS[fixture.model_id]
    profile = hosted_model_config(spec["model"])
    class Model:
        config = SimpleNamespace(max_retries=0, timeout=900, max_connections=1, max_tokens=8192)
        api = SimpleNamespace(client=SimpleNamespace(_client=client, max_retries=0), model_args={"max_retries": 0})
        should_retry = staticmethod(lambda error: False)
        def __str__(self):
            return spec["model"]
        async def generate(self, *, cache=None):
            if cache is not False:
                raise AssertionError("Caching must stay disabled")
            response = await client.post(fixture.core["base_url"] + "/chat/completions", json={
                "model": spec["service_model_id"], "messages": [{"role": "system", "content": "synthetic"},
                    {"role": "user", "content": "synthetic full history"}], "stream": True, "max_tokens": 8192,
                **{key: profile[key] for key in ("temperature", "top_p", "seed")}, **profile["extra_body"]})
            response.raise_for_status()
            if after_http is not None:
                after_http()
            return SimpleNamespace(stop_reason=result_stop, completion='{"action":"synthetic_only"}')
    return Model()


class ContinuityTransferClientTests(unittest.TestCase):
    def setUp(self):
        for name in ("_CONTINUITY_TRANSFER_CORE_REF", "_CONTINUITY_TRANSFER_CORE_RAW",
                     "_CONTINUITY_TRANSFER_CORE_SPECS", "_SECURITY500_CORE_REF", "_PROVIDER352_CORE_REF"):
            p = patch.object(models, name, None)
            p.start(); self.addCleanup(p.stop)
        env = patch.dict(models.os.environ, {}, clear=True)
        env.start(); self.addCleanup(env.stop)
        specs = copy.deepcopy(models.MODEL_SPECS)
        self.addCleanup(lambda: (models.MODEL_SPECS.clear(), models.MODEL_SPECS.update(specs)))

    def test_exact_new_identity_held_and_old_routes_preserved(self):
        spec = models.MODEL_SPECS[FT]
        self.assertEqual(spec["service_model_id"], "sp-lense-continuity-transfer-last8-20261115")
        self.assertEqual(spec["completed_training_updates"], 264)
        self.assertEqual(spec["adapter_sha256"], "f1495fe776512a07ecbdbea70c1e0c517796bf9ca8d96c791893358d028c8d17")
        self.assertFalse(spec["execution_enabled"])
        self.assertEqual(spec["prompt_condition"], "baseline")
        self.assertEqual(models.MODEL_SPECS["sp_lense_security500_neutral"]["service_model_id"],
                         "sp-lense-agent-security500-last8")
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8_neutral"]["adapter_sha256"], models.LAST8_SHA)

    def test_local_alias_retains_identical_generation_profile(self):
        base = hosted_model_config(models.MODEL_SPECS[BASE]["model"])
        ft = hosted_model_config(models.MODEL_SPECS[FT]["model"])
        self.assertEqual({k: v for k, v in base.items() if k != "model"},
                         {k: v for k, v in ft.items() if k != "model"})
        self.assertEqual((ft["temperature"], ft["top_p"], ft["seed"]), (1.0, .95, 20261108))
        self.assertNotIn("thinking_budget", ft["extra_body"])

    def test_held_model_never_builds_or_reads_health_or_legacy_binding(self):
        require_local_files(models.CONTINUITY_TRANSFER_PUBLIC_PATH, models.CONTINUITY_TRANSFER_HANDOFF_PATH)
        with patch.object(models, "build_model", side_effect=AssertionError("model build")), \
                patch.object(models, "_read_local_health", side_effect=AssertionError("health call")):
            for operation in (lambda: models.build_company_model(FT),
                              lambda: models.assert_local_binding(FT),
                              lambda: models.current_local_binding(required_models=[BASE, FT])):
                with self.assertRaisesRegex(ValueError, "HELD"):
                    operation()
            with patch.dict(models.os.environ, {models.CONTINUITY_TRANSFER_CONTEXT_ENV: "not-a-grant"}):
                with self.assertRaises(ValueError):
                    models.build_company_model(BASE)

    def test_public_helper_pin_and_held_template_do_not_qualify(self):
        require_local_files(models.CONTINUITY_TRANSFER_PUBLIC_PATH, models.CONTINUITY_TRANSFER_HANDOFF_PATH,
                            models.CONTINUITY_TRANSFER_HANDOFF_PATH.with_name("SERVICE_CORE_HELD_TEMPLATE_v1.json"))
        h = models.continuity_transfer_handoff_contract()
        self.assertEqual(h.ARMS, ("qwen3.8-27b-base", models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID))
        self.assertEqual(h.COMPLETION_AUDIT_SHA, "1087784cc797f024b5737b200354d2b57bd5dc416d56dbd625fa8711cabc313d")
        raw = models.CONTINUITY_TRANSFER_HANDOFF_PATH.with_name("SERVICE_CORE_HELD_TEMPLATE_v1.json").read_bytes()
        descriptor = {"core_binding": {"path": ".local/synthetic264/held.json", "sha256": hashlib.sha256(raw).hexdigest()},
                      "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}
        with self.assertRaises(ValueError):
            models.initialize_continuity_transfer_core(descriptor, read_bytes=lambda path: raw)
        self.assertIsNone(models.continuity_transfer_planning_core_ref())

    def test_planning_core_never_enables_requests_and_detects_identity_drift(self):
        _, core, raw, descriptor = core_fixture()
        reader = Mock(return_value=raw)
        self.assertEqual(models.initialize_continuity_transfer_core(descriptor, read_bytes=reader), core)
        reader.assert_called_once()
        self.assertEqual(models.selected_models([BASE, FT]), [BASE, FT])
        self.assertFalse(models.MODEL_SPECS[FT].get("execution_enabled", False))
        self.assertEqual(models.current_local_binding(required_models=[BASE, FT]), core)
        with self.assertRaisesRegex(ValueError, "HELD"):
            models.build_company_model(FT)
        with self.assertRaises(ValueError):
            models.initialize_security500_core(descriptor)
        changed = copy.deepcopy(descriptor)
        changed["core_binding"]["path"] = ".local/synthetic264/other.json"
        with self.assertRaisesRegex(ValueError, "immutable"):
            models.initialize_continuity_transfer_core(changed, read_bytes=reader)
        models.MODEL_SPECS[BASE]["max_output_tokens"] = 1024
        with self.assertRaises(ValueError):
            models.selected_models([FT])

    def test_distinct_consumer_holds_before_health_or_metadata_dispatch(self):
        _, _, _, descriptor = core_fixture()
        health, reader = Mock(side_effect=AssertionError("live health")), Mock(side_effect=AssertionError("metadata IO"))
        consumer = models.ContinuityTransferGameConsumer(pins_fixture(), core_binding=descriptor,
            requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4", read_bytes=reader, read_health=health)
        self.assertEqual(consumer.contract.FT, models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID)
        self.assertEqual(consumer._grant_directory(), models.CONTINUITY_TRANSFER_GRANT_DIR)
        with self.assertRaisesRegex(ValueError, "denied"):
            consumer.validate_request(consumer.contract.FT)
        self.assertTrue(consumer._denied)
        reader.assert_called_once(); health.assert_not_called()

    def test_original_case_and_global_clocks_cannot_reset_and_request_uses_minimum(self):
        _, _, _, descriptor = core_fixture()
        clock, monotonic = Mock(return_value=100.0), Mock(return_value=500.0)
        consumer = models.ContinuityTransferGameConsumer(pins_fixture(), core_binding=descriptor,
            requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4", clock=clock, monotonic=monotonic)
        consumer.bind_case_deadline("1970-01-01T00:01:00+00:00", "1970-01-01T04:01:00+00:00")
        expiry = datetime.fromtimestamp(200, timezone.utc).isoformat()
        consumer._bind_global_deadline({"expires_utc": expiry})
        self.assertEqual(consumer._remaining_seconds(), 100)
        monotonic.return_value = 580.0
        clock.return_value = 180.0
        consumer._bind_global_deadline({"expires_utc": expiry})
        self.assertEqual(consumer._remaining_seconds(), 20)
        with self.assertRaises(ValueError):
            consumer._bind_global_deadline({"expires_utc": datetime.fromtimestamp(201, timezone.utc).isoformat()})
        with self.assertRaises(ValueError):
            consumer.bind_case_deadline("1970-01-01T00:01:00+00:00", "1970-01-01T04:01:00+00:00")

    def test_default_health_socket_timeout_uses_remaining_absolute_budget(self):
        _, core, _, descriptor = core_fixture()
        current = [100.0]
        consumer = models.ContinuityTransferGameConsumer(pins_fixture(), core_binding=descriptor,
            requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4",
            clock=lambda: current[0], monotonic=lambda: current[0])
        consumer._case_deadline = 200
        consumer._global_deadline = 300
        consumer._window_expiry = 300
        consumer._request_deadline = 102
        response = Mock()
        response.read.return_value = b'{}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(models.urllib.request, "urlopen", return_value=context) as urlopen:
            self.assertEqual(consumer._read_health(core), {})
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 2)
            def late_read(limit):
                current[0] = 103
                return b'{}'
            response.read.side_effect = late_read
            with self.assertRaisesRegex(ValueError, "deadline exhausted"):
                consumer._read_health(core)

    def test_local_scope_requires_six_paired_neutral_hashes_and_generation_profile(self):
        _, _, _, descriptor = core_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PLAN.json"
            batches = []
            for index, model_id in enumerate((BASE, FT, FT, BASE, BASE, FT)):
                pair = index // 2
                batches.append({"models": [model_id], "seed": 101 + pair, "horizon_hours": 48,
                    "max_decisions": 256, "outer_seconds": 14400, "world_plan_sha256": str(pair + 1) * 64,
                    "initial_user_payload_sha256": str(pair + 1) * 64,
                    "samples": [{"prompt_condition": "baseline", "system_prompt_sha256":
                        "0b52f2098491608007d2bac22a2e9ab2122f5a5c778edd2a348edc7eb98e6ae5"}]})
            frozen = {"protocol": runner.CONTINUITY_TRANSFER_GAME_SCHEMA, "models": [BASE, FT],
                      "generation_seed": 20261108, "generation_profile_sha256": models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
                      "batches": batches}
            path.write_text(json.dumps(frozen), encoding="utf-8")
            consumer = models.ContinuityTransferGameConsumer(pins_fixture(), core_binding=descriptor,
                requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4",
                source_scope={"plan_path": str(path), "batch_index": 0})
            with patch.object(models.Security500GameConsumer, "_validate_game_source_scope", return_value=batches[0]):
                self.assertEqual(consumer._validate_game_source_scope(), batches[0])
                frozen["batches"][1]["initial_user_payload_sha256"] = "9" * 64
                path.write_text(json.dumps(frozen), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Paired"):
                    consumer._validate_game_source_scope()

    @unittest.skipUnless(importlib.util.find_spec("httpx2"), "Established hosted interpreter provides httpx2")
    def test_exact_wire_and_held_guard_with_HTTPMockTransport_no_connections(self):
        import httpx2
        _, core, _, descriptor = core_fixture()
        consumer = models.ContinuityTransferGameConsumer(pins_fixture(), core_binding=descriptor,
            requester_chat="01a17889-f4ee-73a0-813a-5cb7add711d4", read_bytes=Mock(), read_health=Mock())
        spec = models.MODEL_SPECS[FT]
        config = hosted_model_config(spec["model"])
        body = {"messages": [{"role": "user", "content": "synthetic only"}],
                "model": spec["service_model_id"], "max_tokens": 8192, "stream": True,
                **{key: config[key] for key in ("temperature", "top_p", "seed")}, **config["extra_body"]}
        request = httpx2.Request("POST", core["base_url"] + "/chat/completions", json=body)
        consumer._validate_wire(core, request, spec)
        for field, value in (("seed", 1), ("max_tokens", 1024), ("model", models.SECURITY500_SERVICE_MODEL_ID),
                             ("tools", []), ("thinking_budget", 1024)):
            with self.subTest(field=field):
                altered = httpx2.Request("POST", str(request.url), json={**body, field: value})
                with self.assertRaises(ValueError):
                    consumer._validate_wire(core, altered, spec)
        dispatched = []
        async def exercise():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(
                    lambda request: dispatched.append(request) or httpx2.Response(200, json={}))) as client:
                model = SimpleNamespace(api=SimpleNamespace(client=SimpleNamespace(_client=client)))
                with self.assertRaises(ValueError):
                    consumer.install_http_guard(model, FT)
        asyncio.run(exercise())
        self.assertEqual(dispatched, [])


@unittest.skipUnless(importlib.util.find_spec("httpx2"), "Established hosted interpreter provides httpx2")
class ContinuityTransferCompletionBridgeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("inspect_ai") and importlib.util.find_spec("openai"), "Established SDK runtime required")
    def test_actual_Inspect_generate_and_OpenAI_SDK_stream_complete_before_action_return(self):
        import httpx2
        from openai import AsyncOpenAI
        from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, ModelOutput
        fixture, seen = transport_fixture(), []
        base_handler = self.handler(fixture, seen)
        def handler(request):
            base_handler(request)
            route = models.MODEL_SPECS[FT]["service_model_id"]
            chunks = [{"id": "synthetic", "object": "chat.completion.chunk", "created": 100, "model": route,
                       "choices": [{"index": 0, "delta": {"content": "malformed completed text"}, "finish_reason": None}]},
                      {"id": "synthetic", "object": "chat.completion.chunk", "created": 100, "model": route,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
            raw = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx2.Response(200, content=raw.encode(), headers={"content-type": "text/event-stream"})
        class SDKModel:
            def __str__(self):
                return models.MODEL_SPECS[FT]["model"]
            def _resolve_config(self, config):
                return self.config
            async def _prepare_input(self, input, tools, tool_choice, config):
                return SimpleNamespace(input=input)
            async def _generate_prepared(self, prepared, config, cache, on_stream):
                self.observed.append((cache, config.max_retries))
                profile = hosted_model_config(str(self))
                stream = await self.api.client.chat.completions.create(
                    model=models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID,
                    messages=[{"role": "user", "content": "synthetic only"}], stream=True,
                    max_tokens=8192, temperature=profile["temperature"], top_p=profile["top_p"],
                    seed=profile["seed"], extra_body=profile["extra_body"])
                text, finish = "", None
                async for chunk in stream:
                    for choice in chunk.choices:
                        text += choice.delta.content or ""
                        if choice.finish_reason:
                            finish = choice.finish_reason
                return ModelOutput.from_content(models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID, text, stop_reason=finish)
        async def exercise():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
                sdk = AsyncOpenAI(api_key="synthetic-test", organization="synthetic", project="synthetic",
                                  base_url=fixture.core["base_url"], http_client=client, max_retries=2)
                model = SDKModel()
                model.api = SimpleNamespace(client=sdk, model_args={"max_retries": 2})
                model.config = GenerateConfig(max_tokens=8192, timeout=900)
                model.observed = []
                model.generate = types.MethodType(Model.generate, model)
                defaults = dict(sdk.default_headers)
                models._security500_guard_model(model, fixture.consumer, FT)
                result = await model.generate([ChatMessageUser(content="synthetic")])
                self.assertEqual(result.completion, "malformed completed text")
                self.assertEqual(result.stop_reason, "stop")
                self.assertEqual(model.observed, [(False, 0)])
                self.assertEqual(dict(sdk.default_headers), defaults)
                self.assertEqual(fixture.consumer.last_cleanup, "SAME_REQUEST_ACK_KV_EMPTY_BASE_RESTORED")
        asyncio.run(exercise())
        self.assertEqual(len(seen), 1)

    def handler(self, fixture, seen, mutate=None):
        import httpx2
        def handle(request):
            body = models.continuity_transfer_request_parser().parse_request(request.content)
            request_id = request.headers["X-Research4-Request-Id"]
            digest = hashlib.sha256(json.dumps(body["messages"], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            seen.append(request_id)
            evidence = completion_evidence(fixture.consumer.contract, fixture.core, request_id, digest,
                                          body["model"], len(seen))
            evidence["dispatch_record"]["window_sha256"] = fixture.pins["expected_window_sha256"]
            if mutate is not None:
                mutate(evidence)
            fixture.health["last_request_evidence"] = evidence
            return httpx2.Response(200, json={"synthetic": True})
        return handle

    def test_both_routes_return_only_after_exact_request_ack_and_fresh_request_ids(self):
        import httpx2
        for model_id in (BASE, FT):
            with self.subTest(model_id=model_id):
                fixture, seen = transport_fixture(model_id), []
                async def exercise():
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler(fixture, seen))) as client:
                        model = fake_transport_model(fixture, client)
                        fixture.consumer.install_http_guard(model, model_id)
                        for _ in range(2):
                            result = await model.generate()
                            self.assertEqual(result.completion, '{"action":"synthetic_only"}')
                            self.assertEqual(fixture.consumer.last_cleanup, "SAME_REQUEST_ACK_KV_EMPTY_BASE_RESTORED")
                asyncio.run(exercise())
                self.assertEqual(len(seen), 2)
                self.assertNotEqual(seen[0], seen[1])
                self.assertTrue(all(len(value) == 32 and set(value) <= set("0123456789abcdef") for value in seen))
                self.assertEqual(fixture.consumer._attempts, 2)

    def test_wrong_request_or_message_or_seed_tokens_KV_restore_ack_never_returns_action(self):
        import httpx2
        from src.hosted_inspect_runner import IncompleteResponse
        changes = {
            "native_ack_request_id": lambda value: value["completion_ack"].update(request_id="9" * 32),
            "messages": lambda value: value.update(messages_sha256="9" * 64),
            "serviced": lambda value: value.update(model_id_serviced="qwen3.8-27b-base"),
            "seed": lambda value: value.update(seed=123),
            "KV": lambda value: value.update(KV_cache_empty=False),
            "restore": lambda value: value.update(BASE_restored_and_verified=False),
            "token_count": lambda value: value.update(completion_tokens=3),
            "ack": lambda value: value["completion_ack"].update(worker_finished=False),
            "stale": lambda value: value.update(completed_epoch=1.0),
            "EOS_set": lambda value: value.update(eos_token_ids=[9]),
            "EOS_canonical": lambda value: value.update(canonical_eos_token_id=248044),
            "tokenizer_proof": lambda value: value.update(tokenizer_metadata_sha256="9" * 64),
            "vocab": lambda value: value.update(vocab_size=1),
            "token_bounds": lambda value: value.update(actual_generated_token_ids=[248320, 248046]),
        }
        for name, mutate in changes.items():
            with self.subTest(failure=name):
                fixture, seen = transport_fixture(), []
                async def exercise():
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(
                            self.handler(fixture, seen, mutate))) as client:
                        model = fake_transport_model(fixture, client)
                        fixture.consumer.install_http_guard(model, FT)
                        with self.assertRaises(IncompleteResponse):
                            await model.generate()
                        self.assertTrue(client.is_closed)
                        with self.assertRaises(IncompleteResponse):
                            await model.generate()
                asyncio.run(exercise())
                self.assertEqual(len(seen), 1)
                self.assertTrue(fixture.consumer._denied)
                self.assertIn("NO_NEXT_CASE", fixture.consumer.last_cleanup)

    def test_length_result_or_elapsed_case_or_attempt_budget_never_authorizes_action(self):
        import httpx2
        from src.hosted_inspect_runner import IncompleteResponse
        for failure in ("length", "case_deadline", "attempt_budget"):
            with self.subTest(failure=failure):
                fixture, seen = transport_fixture(), []
                if failure == "case_deadline":
                    fixture.consumer._case_deadline = 100.0
                if failure == "attempt_budget":
                    fixture.consumer._attempts = 256
                async def exercise():
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler(fixture, seen))) as client:
                        model = fake_transport_model(fixture, client, result_stop="max_tokens" if failure == "length" else "stop")
                        fixture.consumer.install_http_guard(model, FT)
                        with self.assertRaises(IncompleteResponse):
                            await model.generate()
                asyncio.run(exercise())
                self.assertEqual(len(seen), 1 if failure == "length" else 0)
                self.assertTrue(fixture.consumer._denied)

    def test_timeout_cancels_and_closes_owned_HTTP_and_latches_no_next_case(self):
        import httpx2
        from src.hosted_inspect_runner import IncompleteResponse
        fixture, seen, cancelled = transport_fixture(), [], []
        async def handler(request):
            seen.append(request.headers["X-Research4-Request-Id"])
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        fixture.consumer._remaining_seconds = lambda: .01
        async def exercise():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
                model = fake_transport_model(fixture, client)
                fixture.consumer.install_http_guard(model, FT)
                with self.assertRaises(IncompleteResponse):
                    await model.generate()
                self.assertTrue(client.is_closed)
                with self.assertRaises(IncompleteResponse):
                    await model.generate()
        asyncio.run(exercise())
        self.assertEqual(len(seen), 1)
        self.assertEqual(cancelled, [True])
        self.assertIn("NO_NEXT_CASE", fixture.consumer.last_cleanup)

    def test_valid_post_ACK_cannot_cross_case_global_or_whole_request_deadline(self):
        import httpx2
        from src.hosted_inspect_runner import IncompleteResponse
        for budget in ("case", "global", "request"):
            for stage in ("health_read", "post_validator"):
                with self.subTest(budget=budget, stage=stage):
                    fixture = transport_fixture(global_expiry=101 if budget == "global" else 10000)
                    seen, events, timeouts = [], [], []
                    current = [100.0]
                    consumer = fixture.consumer
                    consumer._clock = lambda: current[0]
                    consumer._monotonic = lambda: current[0]
                    consumer._record_request_evidence = lambda event: events.append(copy.deepcopy(event))
                    if budget == "case":
                        consumer._case_deadline = 101
                    late = 1001.0 if budget == "request" else 102.0
                    def health(core, *, timeout):
                        timeouts.append(timeout)
                        if stage == "health_read" and "last_request_evidence" in fixture.health:
                            current[0] = late
                            fixture.health["updated_epoch"] = late
                            fixture.health["last_request_evidence"]["completed_epoch"] = late
                        return copy.deepcopy(fixture.health)
                    consumer._health_reader = health
                    original_validator = consumer.contract.validate_post_request_metadata
                    def validator(*args, **kwargs):
                        checked = original_validator(*args, **kwargs)
                        if stage == "post_validator":
                            current[0] = late
                        return checked
                    consumer.contract.validate_post_request_metadata = validator
                    async def exercise():
                        async with httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler(fixture, seen))) as client:
                            model = fake_transport_model(fixture, client)
                            consumer.install_http_guard(model, FT)
                            with self.assertRaises(IncompleteResponse):
                                await model.generate()
                            self.assertTrue(client.is_closed)
                            with self.assertRaises(IncompleteResponse):
                                await model.generate()
                    asyncio.run(exercise())
                    self.assertEqual(len(seen), 1)
                    self.assertTrue(consumer._denied)
                    self.assertIsNone(consumer._request_deadline)
                    self.assertIn("NO_NEXT_CASE", consumer.last_cleanup)
                    self.assertTrue(all(0 < value <= 5 for value in timeouts))
                    if stage == "health_read":
                        self.assertEqual(timeouts[-1], 1 if budget in ("case", "global") else 5)
                        late_records = [row for row in events if row["event"] == "HEALTH_READ_AFTER_BUDGET_NO_ACTION"]
                        self.assertEqual(len(late_records), 1)
                        self.assertTrue(late_records[0]["fresh_health"]["last_request_evidence"]["KV_cache_empty"])
                    else:
                        self.assertTrue(any(row["event"] == "POST_REQUEST_VALIDATED" for row in events))

    def test_prevalidation_and_generation_share_one_original_900_second_clock(self):
        import httpx2
        from src.hosted_inspect_runner import IncompleteResponse
        fixture, seen, current, deadlines = transport_fixture(), [], [100.0], []
        consumer = fixture.consumer
        consumer._clock = lambda: current[0]
        consumer._monotonic = lambda: current[0]
        original = consumer._validated_request
        def slow_prevalidation(*args, **kwargs):
            checked = original(*args, **kwargs)
            deadlines.append(consumer._request_deadline)
            current[0] += 400
            fixture.health["updated_epoch"] = current[0]
            return checked
        consumer._validated_request = slow_prevalidation
        async def exercise():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler(fixture, seen))) as client:
                model = fake_transport_model(fixture, client)
                consumer.install_http_guard(model, FT)
                with self.assertRaises(IncompleteResponse):
                    await model.generate()
                self.assertTrue(client.is_closed)
        asyncio.run(exercise())
        self.assertEqual(seen, [])
        self.assertEqual(deadlines, [1000.0, 1000.0, 1000.0])
        self.assertTrue(consumer._denied)


class ContinuityTransferMatrixTests(unittest.TestCase):
    def config(self):
        contrast = [FT, BASE]
        return {"schema": runner.CONTINUITY_TRANSFER_GAME_SCHEMA, "role": "unsealed_development",
            "seeds": [101, 102, 103], "local_models": [BASE, FT], "hosted_models": [],
            "world_version": "company_world_long_horizon_v1", "incentive_profile": "high_reward",
            "horizon_hours": 48, "outer_seconds": 14400, "primary_outcome": "survived_horizon",
            "generation_profile_sha256": models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
            "generation_seed": 20261108,
            "primary_contrast": contrast, "same_prompt_adapter_contrasts": [contrast],
            "comparison_contrasts": [contrast]}

    def test_only_exact_two_neutral_matrix_is_admitted(self):
        self.assertEqual(runner.validate_model_matrix((BASE, FT), ()), ((BASE, FT), ()))
        for local, hosted in (((FT, BASE), ()), ((BASE, FT), ("terra",)),
                              ((BASE, FT, FT), ()), (("qwen_base", FT), ()),
                              ((BASE, "sp_lense_security500_neutral"), ())):
            with self.subTest(local=local, hosted=hosted), self.assertRaises(ValueError):
                runner.validate_model_matrix(local, hosted)

    def test_six_stub_batches_use_existing_rotation_without_instantiating_worlds(self):
        calls = []
        def prospective(seed, *, models, **kwargs):
            calls.append((seed, models))
            return {"seed": seed, "models": models, "horizon_hours": 48, "outer_seconds": 14400,
                    "max_decisions": 256, "samples": [{"case_id": str(seed) + "/" + models[0]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_config.json"
            path.write_text(json.dumps(self.config()), encoding="utf-8")
            with patch.object(runner, "selected_models"), patch.object(runner, "prospective_plan", side_effect=prospective):
                result = runner.plan(path)
            self.assertEqual(result["episodes"], 6)
            self.assertEqual(calls, [(101, [BASE]), (101, [FT]), (102, [FT]),
                                     (102, [BASE]), (103, [BASE]), (103, [FT])])
            self.assertTrue(all(len(batch["models"]) == 1 for batch in result["batches"]))

    def test_wrong_protocol_or_budgets_rejected_before_any_world_creation(self):
        for field, value in (("schema", "old"), ("seeds", list(range(9))), ("horizon_hours", 24),
                             ("outer_seconds", 5400), ("primary_contrast", [BASE, FT])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                config = {**self.config(), field: value}
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(runner, "prospective_plan", side_effect=AssertionError("world creation")):
                    with self.assertRaises(ValueError):
                        runner.plan(path)

    def test_execute_is_held_before_credentials_or_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self.config()), encoding="utf-8")
            destination = Path(tmp) / "not_created"
            with patch.object(runner, "configured_key_names", side_effect=AssertionError("credential read")):
                with self.assertRaisesRegex(ValueError, "HELD"):
                    runner.execute(destination, None, path)
            self.assertFalse(destination.exists())

    def test_six_slot_ledger_keeps_unknown_attempts_and_base_ceiling(self):
        frozen = {"batches": [{"samples": [{"case_id": f"synthetic/{seed}/{model}",
            "model_id": model, "seed": seed}]} for seed in (101, 102, 103) for model in (BASE, FT)]}
        cells = [{"model_id": BASE, "seed": seed, "evidence_status": "audited_complete", "survived_horizon": True}
                 for seed in (101, 102, 103)]
        cells.append({"model_id": FT, "seed": 101, "evidence_status": "incomplete", "survived_horizon": None,
                      "external_stop": "time_limit"})
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            runner.write_json(destination / "PLAN.json", frozen)
            runner.write_continuity_transfer_ledger(destination, frozen, cells,
                attempted_case_ids=[f"synthetic/102/{FT}"])
            ledger = runner.read_json(destination / "CANDIDATE_COHORT_LEDGER.json")
            self.assertEqual(len(ledger["slots"]), 6)
            self.assertFalse(ledger["all_six_audited_known"])
            self.assertEqual(ledger["primary_ceiling_status"], "CEILING_LIMITED")
            self.assertEqual([row["status"] for row in ledger["slots"] if row["model_id"] == FT],
                             ["UNKNOWN_INCOMPLETE", "ATTEMPTED_UNKNOWN_NO_AUDITED_RESULT", "NOT_STARTED_UNKNOWN"])

    def test_child_readback_uses_actual_new_condition_under_new_context(self):
        from src import company_world_evaluation as worker
        with patch.dict(worker.os.environ, {worker.CONTINUITY_TRANSFER_CONTEXT_ENV: "synthetic"}), \
                patch.object(worker, "assert_local_binding", return_value={"ready": True}) as health:
            self.assertEqual(worker.local_binding_readback({"models": [FT]}), {"last8": {"ready": True}})
            health.assert_called_once_with(FT)


if __name__ == "__main__":
    unittest.main()
