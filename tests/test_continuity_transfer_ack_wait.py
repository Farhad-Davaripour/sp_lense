"""Same-request post-SSE ACK wait; synthetic metadata and mock transport only.

Admission/qualification uses the existing bridge fixture. The busy-watch wrapper
is real, with its exact downstream signature bound before the synthetic metadata
checks; the strict post-request validator is the pinned real Root helper.
"""
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import company_models as models
from src.hosted_inspect_runner import IncompleteResponse, hosted_model_config
from tests.test_continuity_transfer_game_client import (
    FT, completion_evidence, fake_transport_model, transport_fixture,
)


@unittest.skipUnless(importlib.util.find_spec("httpx2"), "Established hosted interpreter provides httpx2")
class ContinuityTransferAckWaitTests(unittest.TestCase):
    def scenario(self, *, failure=None, global_expiry=10000):
        import httpx2
        f = transport_fixture(global_expiry=global_expiry)
        f.current = [100.0]
        f.consumer._clock = lambda: f.current[0]
        f.consumer._monotonic = lambda: f.current[0]
        f.requests, f.watch_health, f.watch_kwargs = [], [], []
        f.health_reads, f.wait_delays, f.actions = [], [], []
        f.after_stream = False
        f.current_record = None
        f.failure = failure
        clean = models.continuity_transfer_handoff_contract()
        signature = inspect.signature(clean._validate_bound_metadata)
        decision = copy.deepcopy(f.consumer.contract.validate_per_request_metadata.return_value)

        def bound_metadata(*args, **kwargs):
            # Watch must not receive post-only expected_request_id/hash kwargs.
            signature.bind(*args, **kwargs)
            health = kwargs["fresh_health"]
            for key, value in f.core["expected_health"].items():
                if not clean.same(health.get(key), value):
                    raise clean.RequestDenied("Synthetic watch service identity changed")
            clean.explicit_gate(health)
            if kwargs["validation_stage"] == "WATCH":
                f.watch_health.append(copy.deepcopy(health))
                f.watch_kwargs.append(set(kwargs))
            elif kwargs["validation_stage"] == "IDLE":
                clean.validate_idle_health(health)
            else:
                raise AssertionError("Only watch and strict idle post validation belong to this fixture")
            return copy.deepcopy(decision)

        f.consumer.contract._validate_bound_metadata = Mock(side_effect=bound_metadata)
        watch = f.consumer.contract.validate_busy_watch
        f.consumer.contract.validate_busy_watch = Mock(wraps=watch)
        post = f.consumer.contract.validate_post_request_metadata
        f.consumer.contract.validate_post_request_metadata = Mock(wraps=post)
        original_health = copy.deepcopy(f.health)

        def observation(*, busy, record):
            health = copy.deepcopy(original_health)
            health.update(updated_epoch=f.current[0], pending_http_requests=1 if busy else 0,
                completion_gate={"active_request_id": f.current_record["native_ticket_id"] if busy else None,
                                 "finishing": busy, "unavailable_reason": None},
                last_request_evidence=copy.deepcopy(record))
            return health

        def health_reader(core, *, timeout):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 5)
            if not f.after_stream:
                return copy.deepcopy(original_health)
            number = len(f.health_reads)
            if number == 0:
                health = observation(busy=True, record=None)
            elif failure == "deadline":
                f.current[0] = 104.0
                health = observation(busy=True, record=None)
            elif failure == "revoked_window":
                # Mutation follows one authenticated pending observation. Each
                # next poll must fresh-read the pinned window, not reuse it.
                f.records[".local/synthetic264/window.json"] = b'{"status":"REVOKED"}'
                health = observation(busy=True, record=None)
            elif failure == "service_identity":
                health = observation(busy=True, record=None)
                health["pid"] += 1
            elif failure == "unavailable":
                health = observation(busy=True, record=None)
                health["completion_gate"]["unavailable_reason"] = "synthetic unavailable"
            elif failure == "invalid_current_ack":
                health = observation(busy=True, record=f.current_record)
                health["last_request_evidence"]["completion_ack"]["worker_finished"] = False
            elif number == 1:
                previous = completion_evidence(f.consumer.contract, f.core, "1" * 32, "3" * 64,
                                              f.consumer.contract.BASE)
                previous["completed_epoch"] = 99.0
                previous["dispatch_record"]["window_sha256"] = f.pins["expected_window_sha256"]
                health = observation(busy=False, record=previous)
            elif number == 2:
                health = observation(busy=True, record=f.current_record)
            else:
                health = observation(busy=False, record=f.current_record)
            f.health_reads.append(copy.deepcopy(health))
            return health

        f.consumer._health_reader = health_reader
        if failure == "deadline":
            f.consumer._case_deadline = 103.0

        def handler(request):
            f.requests.append(request)
            body = models.continuity_transfer_request_parser().parse_request(request.content)
            message_hash = hashlib.sha256(json.dumps(body["messages"], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            f.current_record = completion_evidence(f.consumer.contract, f.core,
                request.headers["X-Research4-Request-Id"], message_hash, body["model"])
            f.current_record["dispatch_record"]["window_sha256"] = f.pins["expected_window_sha256"]
            # The actual SDK fixture marks after_stream only after consuming all
            # SSE. The simple bridge marks it at transport return.
            if not getattr(f, "actual_sdk", False):
                f.after_stream = True
            return httpx2.Response(200, json={"synthetic": True})
        f.handler = handler
        return f

    async def fast_wait(self, f, delay, saved_sleep):
        self.assertGreaterEqual(delay, 0)
        self.assertLessEqual(delay, 900)
        f.wait_delays.append(delay)
        await saved_sleep(0)

    @unittest.skipUnless(importlib.util.find_spec("inspect_ai") and importlib.util.find_spec("openai"),
                         "Established Inspect/OpenAI SDK runtime required")
    def test_actual_sdk_stream_waits_for_same_ACK_with_one_POST_and_no_generation_retry(self):
        import httpx2
        from openai import AsyncOpenAI
        from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, ModelOutput
        f = self.scenario()
        f.actual_sdk = True

        def stream_handler(request):
            f.handler(request)
            route = models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID
            chunks = [{"id": "synthetic", "object": "chat.completion.chunk", "created": 100, "model": route,
                       "choices": [{"index": 0, "delta": {"content": '{"action":"synthetic_only"}'}, "finish_reason": None}]},
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
                f.after_stream = True
                return ModelOutput.from_content(models.CONTINUITY_TRANSFER_SERVICE_MODEL_ID, text, stop_reason=finish)

        async def exercise():
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(stream_handler)) as client:
                sdk = AsyncOpenAI(api_key="synthetic-test", organization="synthetic", project="synthetic",
                                  base_url=f.core["base_url"], http_client=client, max_retries=2)
                model = SDKModel()
                model.api = SimpleNamespace(client=sdk, model_args={"max_retries": 2})
                model.config = GenerateConfig(max_tokens=8192, timeout=900)
                model.observed = []
                model.generate = types.MethodType(Model.generate, model)
                models._security500_guard_model(model, f.consumer, FT)
                result = await model.generate([ChatMessageUser(content="synthetic")])
                f.actions.append(result.completion)
                self.assertGreaterEqual(len(f.health_reads), 4)
                self.assertEqual(f.health_reads[-1]["pending_http_requests"], 0)
                self.assertEqual(model.observed, [(False, 0)])
                self.assertEqual(sdk.max_retries, 0)
        saved_sleep = asyncio.sleep
        with patch.object(models.asyncio, "sleep", new=lambda delay: self.fast_wait(f, delay, saved_sleep)):
            asyncio.run(exercise())
        self.assertEqual(len(f.requests), 1)
        self.assertEqual(f.consumer._attempts, 1)
        self.assertEqual(f.actions, ['{"action":"synthetic_only"}'])
        self.assertGreaterEqual(len(f.watch_health), 4)
        self.assertGreaterEqual(len(f.wait_delays), 3)
        self.assertTrue(all("expected_request_id" not in keys and "expected_messages_sha256" not in keys
                            for keys in f.watch_kwargs))
        self.assertEqual(f.consumer.contract.validate_post_request_metadata.call_count, 1)
        self.assertEqual(f.consumer.last_cleanup, "SAME_REQUEST_ACK_KV_EMPTY_BASE_RESTORED")

    def test_pending_permanent_errors_or_deadline_stop_without_action_or_generation_retry(self):
        import httpx2
        for failure in ("deadline", "revoked_window", "service_identity", "unavailable", "invalid_current_ack"):
            with self.subTest(failure=failure):
                f = self.scenario(failure=failure)
                async def exercise():
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(f.handler)) as client:
                        model = fake_transport_model(f, client)
                        f.consumer.install_http_guard(model, FT)
                        with self.assertRaises(IncompleteResponse):
                            result = await model.generate()
                            f.actions.append(result.completion)
                        self.assertTrue(client.is_closed)
                        with self.assertRaises(IncompleteResponse):
                            await model.generate()
                saved_sleep = asyncio.sleep
                with patch.object(models.asyncio, "sleep", new=lambda delay: self.fast_wait(f, delay, saved_sleep)):
                    asyncio.run(exercise())
                self.assertEqual(len(f.requests), 1)
                self.assertEqual(f.consumer._attempts, 1)
                self.assertEqual(f.actions, [])
                self.assertTrue(f.consumer._denied)
                self.assertIn("NO_NEXT_CASE", f.consumer.last_cleanup)
                self.assertLessEqual(len(f.health_reads), 2)

    def test_pending_sleep_is_clipped_to_original_case_global_and_request_budget(self):
        import httpx2
        for budget in ("case", "global", "request"):
            with self.subTest(budget=budget):
                f = self.scenario(global_expiry=101 if budget == "global" else 10000)
                if budget == "case":
                    f.consumer._case_deadline = 100.02
                def almost_out_of_time():
                    f.current[0] = 999.98 if budget == "request" else 100.98 if budget == "global" else 100.0
                    f.current_record["completed_epoch"] = f.current[0]
                saved_sleep = asyncio.sleep
                async def bounded_sleep(delay):
                    remaining = f.consumer._remaining_seconds()
                    self.assertGreater(delay, 0)
                    self.assertLessEqual(delay, .05)
                    self.assertLessEqual(delay, remaining)
                    f.wait_delays.append(delay)
                    f.current[0] += delay + .001
                    await saved_sleep(0)
                async def exercise():
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(f.handler)) as client:
                        model = fake_transport_model(f, client, after_http=almost_out_of_time)
                        f.consumer.install_http_guard(model, FT)
                        with self.assertRaises(IncompleteResponse):
                            await model.generate()
                        self.assertTrue(client.is_closed)
                with patch.object(models.asyncio, "sleep", new=bounded_sleep):
                    asyncio.run(exercise())
                self.assertEqual(len(f.requests), 1)
                self.assertEqual(len(f.health_reads), 1)
                self.assertEqual(len(f.wait_delays), 1)
                self.assertLess(f.wait_delays[0], .021)
                self.assertTrue(f.consumer._denied)
                self.assertEqual(f.actions, [])


if __name__ == "__main__":
    unittest.main()
