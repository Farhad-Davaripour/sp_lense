"""QwenCloud binding and mocked actual Inspect wire tests; no remote requests."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import qwen_cloud as cloud


ENV = {"QWEN_API_KEY": "fixture-not-a-real-key", "QWEN_BASE_URL": cloud.BASE_URL,
       "QWEN_MODEL": "qwen3.8-max"}


class QwenCloudConfigurationTests(unittest.TestCase):
    def test_only_official_https_endpoint_is_accepted(self):
        self.assertEqual(cloud.validate_base_url(cloud.BASE_URL + "/"), cloud.BASE_URL)
        for value in ("http://" + cloud.HOST + "/compatible-mode/v1",
                      "https://example.invalid/compatible-mode/v1",
                      "https://" + cloud.HOST + ".example.invalid/compatible-mode/v1",
                      cloud.BASE_URL + "?key=fixture", cloud.BASE_URL + "#fragment",
                      "https://fixture@" + cloud.HOST + "/compatible-mode/v1",
                      "https://" + cloud.HOST + ":444/compatible-mode/v1",
                      "https://" + cloud.HOST + "/other", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cloud.validate_base_url(value)

    def test_configuration_is_nonsecret_and_does_not_probe_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("\n".join(f"{key}={value}" for key, value in ENV.items()))
            with patch.dict(cloud.os.environ, ENV), patch.object(cloud, "_create_inspect_model") as build:
                safe = cloud.configuration(path, expected_model="qwen3.8-27b")
            self.assertTrue(safe["api_key_configured"])
            self.assertNotIn(ENV["QWEN_API_KEY"], json.dumps(safe))
            self.assertEqual(safe["base_url"], cloud.BASE_URL)
            build.assert_not_called()

    def test_missing_or_conflicting_configuration_never_falls_back(self):
        for updates, selected in (({"QWEN_BASE_URL": "http://127.0.0.1:8769/v1"}, "qwen3.8-27b"),
                                  ({"QWEN_MODEL": "qwen3.8-pro-max"}, "qwen3.8-max"),
                                  ({"QWEN_MODEL": "qwen3.8-max-0902"}, "qwen3.8-max"),
                                  ({"QWEN_API_KEY": ""}, "qwen3.8-27b")):
            with self.subTest(updates=list(updates)), patch.dict(cloud.os.environ, ENV | updates), \
                    patch.object(cloud, "_create_inspect_model") as build:
                with self.assertRaises(ValueError):
                    cloud.build_model(selected)
                build.assert_not_called()

    def test_builder_keeps_cloud_credentials_separate_from_mac(self):
        with patch.dict(cloud.os.environ, ENV | {"RESEARCH4_QWEN_BASE_URL": "local-existing",
                                                 "RESEARCH4_QWEN_API_KEY": "local-unused"}), \
                patch.object(cloud, "_create_inspect_model", return_value="Inspect model") as factory:
            self.assertEqual(cloud.build_model("qwen3.8-27b"), "Inspect model")
            factory.assert_called_once_with("qwen3.8-27b", cloud.BASE_URL, ENV["QWEN_API_KEY"], 8192, 1)
            self.assertEqual(cloud.os.environ["RESEARCH4_QWEN_BASE_URL"], "local-existing")
            self.assertEqual(cloud.os.environ["RESEARCH4_QWEN_API_KEY"], "local-unused")


@unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
class QwenCloudInspectWireTests(unittest.TestCase):
    def test_actual_sdk_sends_total_cap_and_streams_raw_reasoning_for_both_models(self):
        from openai import DefaultAsyncHttpxClient
        import httpx2

        async def exercise(api_model):
            captured, deltas = [], []
            with patch.dict(cloud.os.environ, ENV):
                model = cloud.build_model(api_model)
            self.assertFalse(model.api.http_client.follow_redirects)
            self.assertEqual(str(model), f"qwencloud/{api_model}")
            self.assertNotIn(ENV["QWEN_API_KEY"], json.dumps(model.model_args))
            await model.api.aclose()

            async def handler(request):
                self.assertEqual(request.url.host, cloud.HOST)
                self.assertEqual(request.url.scheme, "https")
                self.assertEqual(request.url.path, "/compatible-mode/v1/chat/completions")
                body = json.loads(request.content)
                captured.append(body)
                chunks = [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "fixture reasoning"}},
                    {"index": 0, "delta": {"content": '{"action":"check_status"}'}},
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ]
                stream = "".join("data: " + json.dumps({"id": "fixture", "object": "chat.completion.chunk",
                    "created": 1, "model": api_model, "choices": [chunk]}) + "\n\n" for chunk in chunks)
                return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                       content=stream + "data: [DONE]\n\n")

            model.api.http_client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler),
                                                            follow_redirects=False, trust_env=False)
            model.api.initialize()
            try:
                async def observe(event):
                    deltas.append(event)
                output = await model.generate("Return one JSON action.", on_stream=observe)
                self.assertEqual(output.completion, '{"action":"check_status"}')
                self.assertEqual(len(captured), 1)
                wire = captured[0]
                self.assertEqual(wire["model"], api_model)
                self.assertEqual(wire["max_completion_tokens"], 8192)
                self.assertNotIn("max_tokens", wire)
                self.assertNotIn("thinking_budget", wire)
                self.assertNotIn("tools", wire)
                self.assertTrue(wire["enable_thinking"])
                self.assertEqual(wire["reasoning_effort"], "low")
                self.assertTrue(wire["stream"])
                self.assertTrue(any(event.type == "reasoning" and event.reasoning == "fixture reasoning"
                                    for event in deltas))
            finally:
                await model.api.aclose()

        for api_model in ("qwen3.8-max", "qwen3.8-27b"):
            with self.subTest(model=api_model):
                asyncio.run(exercise(api_model))

    def test_redirect_is_not_followed(self):
        from openai import DefaultAsyncHttpxClient
        import httpx2

        async def exercise():
            requests = []
            with patch.dict(cloud.os.environ, ENV):
                model = cloud.build_model("qwen3.8-max")
            await model.api.aclose()
            async def handler(request):
                requests.append(str(request.url))
                return httpx2.Response(302, headers={"location": "https://example.invalid/redirect"})
            model.api.http_client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler),
                                                            follow_redirects=False, trust_env=False)
            model.api.initialize()
            try:
                with self.assertRaises(Exception):
                    await model.generate("This is a mocked request.")
                self.assertEqual(len(requests), 1)
                self.assertTrue(requests[0].startswith(cloud.BASE_URL))
            finally:
                await model.api.aclose()
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
