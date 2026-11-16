"""Hosted Inspect/OpenAI SDK compatibility using only synthetic mock transport.

Run with the established hosted Python. No Inspect Model, model weights, native
backend, credential file or external connection is created. Windows asyncio's
internal local socketpair needs a narrowly scoped loopback connection.
"""
import asyncio
import builtins
import importlib.util
import pathlib
import socket
import threading
import types
import unittest
import urllib.request
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch


class HostedSDKCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("inspect_ai", "openai", "httpx2")),
                         "Hosted Inspect/OpenAI SDK dependencies are unavailable")
    def test_actual_sdk_model_method_and_mock_transport(self):
        original_import, original_open = builtins.__import__, builtins.open
        original_path_open, original_connect, original_pair = pathlib.Path.open, socket.socket.connect, socket.socketpair
        pair_state = threading.local()

        def safe_import(name, *args, **kwargs):
            if name.partition(".")[0] in {"mlx", "mlx_lm", "torch", "transformers", "tensorflow", "jax", "vllm"}:
                raise AssertionError("Native/model import blocked")
            return original_import(name, *args, **kwargs)

        def safe_open(file, *args, **kwargs):
            if isinstance(file, (str, pathlib.Path)) and pathlib.Path(file).name.lower() == ".env":
                raise AssertionError(".env access blocked")
            return original_open(file, *args, **kwargs)

        def safe_path_open(path, *args, **kwargs):
            if path.name.lower() == ".env":
                raise AssertionError(".env access blocked")
            return original_path_open(path, *args, **kwargs)

        def blocked(*args, **kwargs):
            raise AssertionError("External network blocked")

        def connect(sock, address):
            if (getattr(pair_state, "creating", False) and isinstance(address, tuple)
                    and address[0] == "127.0.0.1"):
                return original_connect(sock, address)
            return blocked()

        def socketpair(*args, **kwargs):
            pair_state.creating = True
            try:
                return original_pair(*args, **kwargs)
            finally:
                pair_state.creating = False

        with ExitStack() as stack:
            for target, name, value in (
                (builtins, "__import__", safe_import), (builtins, "open", safe_open),
                (pathlib.Path, "open", safe_path_open), (socket.socket, "connect", connect),
                (socket.socket, "connect_ex", blocked), (socket, "socketpair", socketpair),
                (socket, "create_connection", blocked), (urllib.request, "urlopen", blocked),
            ):
                stack.enter_context(patch.object(target, name, value))

            import httpx2
            from inspect_ai.model import ChatMessageUser, GenerateConfig, Model
            from inspect_ai.model import _retry as inspect_retry
            from openai import AsyncOpenAI, InternalServerError
            from tenacity import retry
            from src import company_models as models
            from src.hosted_inspect_runner import IncompleteResponse, hosted_model_config
            from tests.test_security500_client import Security500ClientTests

            self.assertEqual(hosted_model_config(models.MODEL_SPECS["terra"]["model"])["reasoning_effort"], "low")
            self.assertEqual(hosted_model_config(models.MODEL_SPECS["deepseek"]["model"])["extra_body"]["thinking"],
                             {"type": "enabled"})

            class FakeModel:
                def __str__(self):
                    return models.MODEL_SPECS[self.name]["model"]

                def _resolve_config(self, config):
                    return self.config

                async def _prepare_input(self, input, tools, tool_choice, config):
                    return SimpleNamespace(input=input)

                async def _generate_prepared(self, prepared, config, cache, on_stream):
                    self.observed.append((cache, config.max_retries))
                    if self.name == "terra":
                        await self.api.client.responses.create(model="gpt-6.1-terra", input="synthetic", stream=True)
                    elif self.name == "deepseek":
                        await self.api.client.chat.completions.create(
                            model="deepseek-flash", messages=[{"role": "user", "content": "synthetic"}], stream=True)
                    else:
                        spec = models.MODEL_SPECS[self.name]
                        profile = hosted_model_config(spec["model"])
                        await self.api.client.chat.completions.create(
                            model=spec["service_model_id"], messages=[{"role": "user", "content": "synthetic"}],
                            stream=True, max_tokens=8192, temperature=profile["temperature"],
                            top_p=profile["top_p"], seed=profile["seed"], extra_body=profile["extra_body"])
                    return "mock transport"

            async def exercise(name, status):
                fixture = Security500ClientTests(methodName="test_cached_response_without_http_is_rejected")
                fixture.setUp()
                try:
                    context = fixture.make_consumer()
                    context.consumer._child_plan_verified = True
                    context.consumer._validate_game_source_scope = lambda: {"models": [name]}
                    requests = []

                    async def handler(request):
                        requests.append(request)
                        if status == "redirect":
                            return httpx2.Response(307, headers={"location": "https://example.invalid/leak"})
                        return httpx2.Response(
                            status, content=b"data: [DONE]\n\n" if status == 200 else b'{"error":{"message":"synthetic"}}',
                            headers={"content-type": "text/event-stream" if status == 200 else "application/json"})

                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler),
                                                  follow_redirects=True) as client:
                        sdk = AsyncOpenAI(api_key="dummy-test", organization="dummy", project="dummy",
                            base_url="http://127.0.0.1:8769/v1" if name.startswith(("qwen", "sp_lense"))
                            else "https://example.invalid/v1", http_client=client, max_retries=2)
                        default_headers = dict(sdk.default_headers)
                        model = FakeModel()
                        model.name = name
                        model.api = SimpleNamespace(client=sdk, model_args={"max_retries": 2})
                        model.config = GenerateConfig(max_tokens=8192)
                        model.observed = []
                        # Exercise the actual SDK method without constructing an Inspect Model.
                        model.generate = types.MethodType(Model.generate, model)
                        models._security500_guard_model(model, context.consumer, name)
                        self.assertEqual((sdk.max_retries, model.config.max_retries, model.config.cache), (0, 0, False))
                        attempts, sleeps = [], []

                        async def forbidden_sleep(seconds):
                            sleeps.append(seconds)
                            raise AssertionError("Retry backoff must not occur")

                        retry_config = inspect_retry.model_retry_config(
                            "synthetic", 0, None, model.should_retry, lambda error: None,
                            lambda model_name, state: None)
                        retry_config["sleep"] = forbidden_sleep

                        @retry(**retry_config, reraise=True)
                        async def retryable_failure():
                            attempts.append(1)
                            raise RuntimeError("synthetic retryable failure")

                        # A live override permits retries, but the cohort's actual
                        # retry predicate prevents a retry or backoff entirely.
                        with patch.object(inspect_retry, "generate_config_override",
                                          side_effect=lambda key, value: 9 if key == "max_retries" else value):
                            with self.assertRaisesRegex(RuntimeError, "synthetic retryable failure"):
                                await retryable_failure()
                        self.assertEqual((len(attempts), sleeps), (1, []))
                        message = [ChatMessageUser(content="synthetic")]
                        if status == "config_override":
                            with self.assertRaises(IncompleteResponse):
                                await model.generate(message, config=GenerateConfig(max_retries=9))
                            self.assertEqual(requests, [])
                        elif status == "redirect":
                            with self.assertRaises(IncompleteResponse):
                                await model.generate(message)
                            self.assertEqual(len(requests), 1)
                            self.assertEqual(str(requests[0].url), "http://127.0.0.1:8769/v1/chat/completions")
                        elif status == 200:
                            self.assertEqual(await model.generate(message), "mock transport")
                            self.assertEqual(await model.generate(message), "mock transport")
                            self.assertEqual(model.observed, [(False, 0), (False, 0)])
                            self.assertEqual(len(requests), 2)
                            with self.assertRaises(IncompleteResponse):
                                await model.generate(message, cache=True)
                            self.assertEqual(len(requests), 2)
                        else:
                            with self.assertRaises(InternalServerError):
                                await model.generate(message)
                            self.assertEqual(model.observed, [(False, 0)])
                            self.assertEqual(len(requests), 1)
                        self.assertEqual(dict(sdk.default_headers), default_headers)
                        for request in requests:
                            root_headers = {key.lower(): value for key, value in request.headers.multi_items()
                                            if key.lower().startswith("x-research4-")}
                            if name.startswith(("qwen", "sp_lense")):
                                self.assertEqual(root_headers, {
                                    "x-research4-requester": context.handoff.GAME_CHAT,
                                    "x-research4-grant-nonce": context.window["grant_nonce"],
                                    "x-research4-game-plan-sha256": context.window["game_plan_sha256"],
                                    "x-research4-window-sha256": context.pins["expected_window_sha256"],
                                })
                            else:
                                self.assertEqual(root_headers, {})
                finally:
                    fixture.doCleanups()

            async def run():
                for name in ("sp_lense_security500_neutral", "sp_lense_security500", "terra", "deepseek"):
                    await exercise(name, 200)
                for name in ("sp_lense_security500_neutral", "terra", "deepseek"):
                    await exercise(name, 500)
                await exercise("sp_lense_security500_neutral", "config_override")
                await exercise("sp_lense_security500", "redirect")

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
