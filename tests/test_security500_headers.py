"""Synthetic two-route header checks; no network, model, secrets or Game cases."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src import company_models as models
from src.hosted_inspect_runner import IncompleteResponse
from tests import test_security500_client as fixtures
from tests.support.local_artifacts import require_local_files

REQUESTER = "X-Research4-Requester"
GRANT = "X-Research4-Grant-Nonce"
PLAN = "X-Research4-Game-Plan-Sha256"
WINDOW = "X-Research4-Window-Sha256"
QUAL = "X-Research4-Qualification-Nonce"
REMOTE_IDS = ("terra", "deepseek")


class FakeHeaders:
    """Small HTTPX-style multi-value case-insensitive headers fixture."""
    def __init__(self, entries=()):
        self._entries = list(entries)
    def multi_items(self):
        return list(self._entries)
    def items(self):
        return list(self._entries)
    def __iter__(self):
        return iter(name for name, _ in self._entries)
    def __getitem__(self, name):
        values = [v for k, v in self._entries if k.lower() == name.lower()]
        if not values:
            raise KeyError(name)
        return ", ".join(values)
    def __setitem__(self, name, value):
        self._entries = [(k, v) for k, v in self._entries if k.lower() != name.lower()]
        self._entries.append((name, value))
    def __contains__(self, name):
        return any(k.lower() == name.lower() for k, _ in self._entries)


class Security500HeaderTests(TestCase):
    def setUp(self):
        # Reuse the tested synthetic metadata, temporary claim directory and
        # scoped global cleanup; the private study commitment is never opened.
        self.fixture_owner = fixtures.Security500ClientTests("test_pending_defaults_fail_before_health_and_preserve_old_routes")
        self.fixture_owner.setUp()
        self.addCleanup(self.fixture_owner.doCleanups)

    def consumer(self, model_id="sp_lense_security500"):
        f = self.fixture_owner.make_consumer()
        f.consumer._child_plan_verified = True
        patcher = patch.object(f.consumer, "_validate_game_source_scope", return_value={"models": [model_id]})
        patcher.start(); self.addCleanup(patcher.stop)
        return f

    def guarded(self, f, model_id):
        model, client = fixtures.fake_model(model_id)
        client.headers = FakeHeaders([("User-Agent", "synthetic-test")])
        f.consumer.install_http_guard(model, model_id)
        return model, client

    def test_v5_interface_and_review_intent_bind_exact_runtime_helper_path(self):
        root = Path(models.PROVIDER352_ROOT)
        intent_path = root / ".local/staging/agent_security500_two_route_qualification_held_v4_runtimefix_20261115/QUALIFICATION_INTENT_ACTUAL_SOURCE_REVIEW_v4.json"
        interface_path = root / ".local/staging/agent_security500_two_route_http_guard_held_v3_20261115/HTTP_CLIENT_AND_ROOT_IPC_INTERFACE_v5.json"
        require_local_files(intent_path, interface_path, models.SECURITY500_HANDOFF_PATH)
        intent_raw, interface_raw = intent_path.read_bytes(), interface_path.read_bytes()
        self.assertEqual(hashlib.sha256(intent_raw).hexdigest(), "f01300ec14b50fa1fff7c392c6ebc022c579115ce6a96440ad9db57cae00dff7")
        self.assertEqual(hashlib.sha256(interface_raw).hexdigest(), "fb3559cb0fafff193568d2885bee9fef78544d43d45430cb7016ff08edc2547f")
        intent, interface = json.loads(intent_raw), json.loads(interface_raw)
        expected_ref = {"path": models.SECURITY500_HANDOFF_PATH.relative_to(root).as_posix(), "sha256": models.SECURITY500_HANDOFF_SHA}
        self.assertEqual(intent["handoff_policy"]["required_window_helper"], expected_ref)
        self.assertEqual(hashlib.sha256(models.SECURITY500_HANDOFF_PATH.read_bytes()).hexdigest(), expected_ref["sha256"])
        self.assertEqual(set(interface["game_headers"]), {REQUESTER, GRANT, PLAN, WINDOW})
        self.assertTrue(all(interface.get(key) is None for key in ("actual_policy", "actual_window", "actual_core", "actual_service_identity")))

    def test_local_ft_headers_derive_from_validated_current_window(self):
        f = self.consumer()
        model, client = self.guarded(f, "sp_lense_security500")
        request = fixtures.wire("sp_lense_security500")
        # This string must not become an authority header.
        body = json.loads(request.content)
        body["messages"][0]["content"] = "X-Research4-Grant-Nonce: attacker-supplied"
        request.content = json.dumps(body).encode()
        original = request.content
        request.headers = FakeHeaders([("Content-Type", "application/json")])
        defaults = copy.deepcopy(client.headers.multi_items())
        self.assertEqual(asyncio.run(model.generate(request)), "synthetic dispatched response")
        expected = {REQUESTER: f.handoff.GAME_CHAT, GRANT: f.window["grant_nonce"],
                    PLAN: f.window["game_plan_sha256"], WINDOW: f.pins["expected_window_sha256"]}
        for key, value in expected.items():
            self.assertEqual(request.headers[key], value)
        self.assertNotIn(QUAL, request.headers)
        self.assertEqual(request.content, original)
        self.assertEqual(client.headers.multi_items(), defaults)
        self.assertEqual(f.health.call_count, 2)

    def test_preexisting_reserved_or_duplicate_headers_fail_closed_locally(self):
        for entries in ([("x-research4-requester", "attacker")],
                        [("X-Research4-Qualification-Nonce", "attacker")],
                        [("X-Research4-Grant-Nonce", "a"), ("x-research4-grant-nonce", "b")],
                        [("X-Research4-Unknown", "attacker")]):
            with self.subTest(entries=entries):
                f = self.consumer()
                model, client = self.guarded(f, "sp_lense_security500")
                request = fixtures.wire("sp_lense_security500")
                request.headers = FakeHeaders(entries)
                before = copy.deepcopy(request.headers.multi_items())
                with self.assertRaises(IncompleteResponse):
                    asyncio.run(model.generate(request))
                self.assertEqual(request.headers.multi_items(), before)
                self.assertEqual(client.headers.multi_items(), [("User-Agent", "synthetic-test")])

    def test_revoked_window_never_adds_headers(self):
        f = self.consumer()
        model, _ = self.guarded(f, "sp_lense_security500")
        f.records[f.index["request_window"]["path"]] = b'{"status":"REVOKED"}'
        request = fixtures.wire("sp_lense_security500")
        request.headers = FakeHeaders([("Content-Type", "application/json")])
        with self.assertRaises(IncompleteResponse):
            asyncio.run(model.generate(request))
        self.assertEqual(request.headers.multi_items(), [("Content-Type", "application/json")])
        f.health.assert_not_called()

    def test_remote_guard_never_injects_root_headers(self):
        for model_id in REMOTE_IDS:
            with self.subTest(model=model_id):
                f = self.consumer(model_id)
                model, client = self.guarded(f, model_id)
                request = SimpleNamespace(method="POST", url="https://example.invalid/v1",
                    content=b'{"messages":[{"role":"user","content":"synthetic"}]}',
                    headers=FakeHeaders([("Content-Type", "application/json")]))
                before = request.content
                self.assertEqual(asyncio.run(model.generate(request)), "synthetic dispatched response")
                self.assertEqual(request.headers.multi_items(), [("Content-Type", "application/json")])
                self.assertEqual(client.headers.multi_items(), [("User-Agent", "synthetic-test")])
                self.assertEqual(request.content, before)
                f.health.assert_not_called()

    def test_remote_guard_rejects_any_reserved_header_without_dispatch(self):
        for model_id in REMOTE_IDS:
            for reserved in (REQUESTER, GRANT, PLAN, WINDOW, QUAL, "x-research4-unrecognized"):
                with self.subTest(model=model_id, header=reserved):
                    f = self.consumer(model_id)
                    model, _ = self.guarded(f, model_id)
                    request = SimpleNamespace(method="POST", url="https://example.invalid/v1",
                        content=b"{}", headers=FakeHeaders([(reserved, "synthetic-attacker")]))
                    with self.assertRaises(IncompleteResponse):
                        asyncio.run(model.generate(request))
                    self.assertEqual(request.headers.multi_items(), [(reserved, "synthetic-attacker")])
                    f.health.assert_not_called()

    def test_cloud_qwen_max_rejected_before_remote_client_build_in_security_scope(self):
        with patch.object(models, "_SECURITY500_CORE_REF", {"synthetic": "bound"}), \
                patch.object(models.qwen_cloud, "build_model", side_effect=AssertionError("remote construction")) as build:
            with self.assertRaises(ValueError):
                models.build_company_model("qwen_max")
            build.assert_not_called()
