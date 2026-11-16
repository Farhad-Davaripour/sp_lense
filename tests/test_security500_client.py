"""Synthetic two-route client tests; no network, model, weights or Game cases."""
import ast
import asyncio
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from src import company_models as models
from src.hosted_inspect_runner import hosted_model_config


ROOT = Path("C:/Users/farha/repos/sp_lense4/research4")
PACKET = ROOT / ".local/staging/agent_security500_two_route_qualification_held_v4_runtimefix_20261115"
HELPER_SHA = "800366ff119b3523260a9f86ccd2ab0eb2d65c2ea30ed81038fb16251d9b68e4"
TEST_SHA = "789744350397a3b070944bac2b890256ce9ce23d4ec6f4a93f4f101023e94021"
GAME_CHAT = "01a17889-f4ee-73a0-813a-5cb7add711d4"
FT_NAMES = ("sp_lense_security500_neutral", "sp_lense_security500")


def synthetic_fixture(*, progress=None):
    """Execute just the reviewed packet's pure fixture functions, not its runner."""
    helper_path = PACKET / "two_route_handoff_contract.py"
    source = helper_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location("security500_test_handoff", helper_path)
    handoff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handoff)
    test_path = PACKET / "test_two_route_outer_cpu.py"
    test_source = test_path.read_bytes()
    assert hashlib.sha256(test_source).hexdigest() == TEST_SHA
    tree = ast.parse(test_source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"bytes_of", "digest", "bound", "fixture"}]
    namespace = {"json": json, "hashlib": hashlib, "handoff": handoff}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(test_path), "exec"), namespace)
    raw, args = namespace["fixture"]()
    if progress is None:
        progress = {"phase": "complete", "verified_updates": 500, "total_updates": 500,
                    "updated_epoch": 1700000000.0}
    encode, digest, bound = (namespace[name] for name in ("bytes_of", "digest", "bound"))
    raw["progress"] = encode(progress)
    core, index, window = (json.loads(raw[k]) for k in ("core", "index", "window"))
    core["artifacts"]["progress"] = bound("progress", raw["progress"])
    raw["core"] = encode(core)
    window["core_binding"] = bound("core", raw["core"])
    window["bound_proofs"] = handoff.linked_proofs(core)
    raw["window"] = encode(window)
    index["core_binding"] = bound("core", raw["core"])
    index["request_window"] = bound("window", raw["window"])
    raw["index"] = encode(index)
    args.update(trusted_core_sha256=digest(raw["core"]),
                trusted_window_sha256=digest(raw["window"]),
                trusted_index_sha256=digest(raw["index"]))
    core, index, window = (json.loads(raw[k]) for k in ("core", "index", "window"))
    records = {
        index["core_binding"]["path"]: raw["core"],
        index["request_window"]["path"]: raw["window"],
        core["artifacts"]["completion_audit"]["path"]: raw["audit"],
        core["artifacts"]["completion_auditor_review"]["path"]: raw["peer"],
        core["artifacts"]["progress"]["path"]: raw["progress"],
        core["readiness_receipt"]["path"]: raw["ready"],
        core["two_route_qualification"]["path"]: raw["cert"],
        ".local/tests/index.json": raw["index"],
    }
    pins = {"index_path": ".local/tests/index.json",
            "expected_index_sha256": args["trusted_index_sha256"],
            "expected_window_sha256": args["trusted_window_sha256"],
            "expected_nonce": args["trusted_nonce"],
            "expected_game_plan_sha256": args["trusted_game_plan_sha256"],
            "expected_direct_authority_sha256": args["trusted_direct_authority_sha256"]}
    descriptor = {"core_binding": index["core_binding"],
                  "trusted_review_sha256": args["trusted_review_sha256"]}
    reads = []
    def read_bytes(path):
        relative = Path(path).relative_to(ROOT.resolve()).as_posix()
        reads.append(relative)
        return records[relative]
    health = Mock(side_effect=lambda binding: copy.deepcopy(args["fresh_health"]))
    clock = Mock(return_value=100.0)
    return SimpleNamespace(handoff=handoff, raw=raw, args=args, core=core,
        index=index, window=window, records=records, pins=pins,
        descriptor=descriptor, reads=reads, read_bytes=read_bytes, health=health, clock=clock)


def fake_model(model_id):
    spec = models.MODEL_SPECS[model_id]
    client = SimpleNamespace(event_hooks={"request": [], "response": []})
    class FakeModel:
        api = SimpleNamespace(client=SimpleNamespace(_client=client, max_retries=0), model_args={"max_retries": 0})
        config = SimpleNamespace(max_retries=0, max_tokens=8192, max_connections=1)
        should_retry = staticmethod(lambda error: False)
        def __str__(self):
            return spec["model"]
        async def generate(self, request, *, cache=None):
            if cache is not False:
                raise AssertionError("Response caching must be explicitly disabled")
            for hook in client.event_hooks["request"]:
                await hook(request)
            return "synthetic dispatched response"
    return FakeModel(), client


def wire(model_id):
    spec = models.MODEL_SPECS[model_id]
    profile = hosted_model_config(spec["model"])
    body = {key: profile[key] for key in ("temperature", "top_p", "seed")}
    body.update(profile["extra_body"], model=spec["service_model_id"],
                max_tokens=8192, stream=True,
                messages=[{"role": "user", "content": "synthetic fixture"}])
    return SimpleNamespace(method="POST", url="http://127.0.0.1:8769/v1/chat/completions",
                           content=json.dumps(body).encode(), headers={})


class Security500ClientTests(TestCase):
    def setUp(self):
        # Protect the shared registry/global planning state from other tests.
        for name, value in (("SECURITY500_HANDOFF_PATH", PACKET / "two_route_handoff_contract.py"),
                            ("SECURITY500_HANDOFF_SHA", HELPER_SHA)):
            item = patch.object(models, name, value)
            item.start(); self.addCleanup(item.stop)
        repo = Path(models.__file__).resolve().parents[1]
        grant_temp = tempfile.TemporaryDirectory(prefix="security500_grant_test_", dir=repo / ".local")
        self.addCleanup(grant_temp.cleanup)
        grants = Path(grant_temp.name) / "grants"
        grants.mkdir()
        grant_patch = patch.object(models, "SECURITY500_GRANT_DIR", grants)
        grant_patch.start(); self.addCleanup(grant_patch.stop)
        for name in ("_SECURITY500_CORE_REF", "_SECURITY500_CORE_RAW", "_SECURITY500_CORE_SPECS",
                     "_SECURITY500_ACTIVE_CONTEXT", "_SECURITY500_ACTIVE_CONSUMER"):
            if hasattr(models, name):
                item = patch.object(models, name, None)
                item.start(); self.addCleanup(item.stop)
        for name in FT_NAMES:
            item = patch.dict(models.MODEL_SPECS[name], copy.deepcopy(models.MODEL_SPECS[name]), clear=True)
            item.start(); self.addCleanup(item.stop)

    def test_pending_defaults_fail_before_health_and_preserve_old_routes(self):
        self.assertIsNone(models.security500_planning_core_ref())
        for name in FT_NAMES:
            spec = models.MODEL_SPECS[name]
            self.assertIsNone(spec.get("adapter_sha256"))
            self.assertIsNone(spec.get("adapter_config_sha256"))
            with patch.object(models.urllib.request, "urlopen", side_effect=AssertionError("network")):
                with self.assertRaises(ValueError):
                    models.selected_models([name])
        self.assertEqual(models.MODEL_SPECS["sp_lense_last8"]["adapter_sha256"], models.LAST8_SHA)
        self.assertEqual(models.MODEL_SPECS["sp_lense_retrained"]["adapter_sha256"], models.RETRAINED_SHA)

    def test_neutral_and_icl_share_exact_new_route_and_native_profile(self):
        neutral, icl = (models.MODEL_SPECS[name] for name in FT_NAMES)
        self.assertEqual(neutral["service_model_id"], "sp-lense-agent-security500-last8")
        self.assertEqual(neutral["model"], icl["model"])
        self.assertEqual(neutral["service_model_id"], icl["service_model_id"])
        self.assertEqual(neutral["prompt_condition"], "baseline")
        self.assertEqual(icl["prompt_condition"], "security_maximization")
        config = hosted_model_config(neutral["model"])
        self.assertEqual((config["temperature"], config["top_p"], config["seed"]), (1.0, .95, 20261108))
        self.assertEqual(config["extra_body"]["chat_template_kwargs"],
                         {"enable_thinking": True, "reasoning_effort": "low"})

    def test_planning_label_uses_hash_bound_completed_progress(self):
        f = synthetic_fixture()
        models.initialize_security500_core(f.descriptor, read_bytes=f.read_bytes)
        timestamp = datetime.fromtimestamp(1700000000.0, timezone.utc) - timedelta(hours=6)
        expected = "SP Lens - " + timestamp.strftime("%Y-%m-%d %H:%M") + " ABT"
        for name in FT_NAMES:
            self.assertIn(expected, models.MODEL_SPECS[name]["label"])
            self.assertEqual(models.MODEL_SPECS[name]["security500_readiness"], "PLANNING_ONLY")

    def test_incomplete_or_invalid_progress_rejected_before_spec_mutation(self):
        original = {name: copy.deepcopy(models.MODEL_SPECS[name]) for name in FT_NAMES}
        for progress in ({"phase": "running", "verified_updates": 500, "total_updates": 500, "updated_epoch": 1700000000.0},
                         {"phase": "complete", "verified_updates": 499, "total_updates": 500, "updated_epoch": 1700000000.0},
                         {"phase": "complete", "verified_updates": 500, "total_updates": 500, "updated_epoch": 0.0},
                         {"phase": "complete", "verified_updates": 500, "total_updates": 500, "updated_epoch": True},
                         {"phase": "complete", "verified_updates": 500, "total_updates": 500, "updated_epoch": float("inf")},
                         {"phase": "complete", "verified_updates": 500, "total_updates": 500, "updated_epoch": 1e30},
                         {"phase": "complete", "verified_updates": 500, "total_updates": 500}):
            with self.subTest(progress=progress):
                f = synthetic_fixture(progress=progress)
                with self.assertRaises(ValueError):
                    models.initialize_security500_core(f.descriptor, read_bytes=f.read_bytes)
                self.assertIsNone(models.security500_planning_core_ref())
                self.assertEqual({name: models.MODEL_SPECS[name] for name in FT_NAMES}, original)

    def test_changed_progress_bytes_reject_before_spec_mutation(self):
        f = synthetic_fixture()
        original = {name: copy.deepcopy(models.MODEL_SPECS[name]) for name in FT_NAMES}
        f.records[f.core["artifacts"]["progress"]["path"]] = b'{"phase":"complete"}'
        with self.assertRaises(ValueError):
            models.initialize_security500_core(f.descriptor, read_bytes=f.read_bytes)
        self.assertIsNone(models.security500_planning_core_ref())
        self.assertEqual({name: models.MODEL_SPECS[name] for name in FT_NAMES}, original)

    def make_consumer(self):
        fixture = synthetic_fixture()
        models.initialize_security500_core(fixture.descriptor, read_bytes=fixture.read_bytes)
        consumer = models.Security500GameConsumer(fixture.pins, core_binding=fixture.descriptor,
            requester_chat=GAME_CHAT, read_bytes=fixture.read_bytes,
            read_health=fixture.health, clock=fixture.clock)
        fixture.consumer = consumer
        return fixture

    def mark_wire_fixture(self, consumer):
        """Isolate transport checks from the separate real plan-scope tests."""
        consumer._child_plan_verified = True
        mock = patch.object(consumer, "_validate_game_source_scope",
                            return_value={"models": ["sp_lense_security500_neutral", "sp_lense_security500"]})
        mock.start(); self.addCleanup(mock.stop)

    def source_scoped_fixture(self):
        """Real local plan/config bytes, synthetic batch; no private study inputs."""
        f = self.make_consumer()
        repo = Path(models.__file__).resolve().parents[1]
        temp = tempfile.TemporaryDirectory(prefix="security500_client_test_", dir=repo / ".local")
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        config = directory / "synthetic_config.yaml"
        config.write_text("synthetic: true\n", encoding="utf-8")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        batch = {"models": ["sp_lense_security500_neutral"], "source_sha256": {"synthetic": "0" * 64},
                 "seed": 0, "world_plan_sha256": "1" * 64, "rules_version": "synthetic",
                 "incentive_profile": "synthetic", "generation_settings": {}, "samples": []}
        frozen = {"configuration_path": str(config), "configuration_sha256": digest(config),
                  "orchestrator_source_sha256": digest(repo / "src/infrastructure_preservation_comparison.py"),
                  "batches": [batch]}
        plan_path = directory / "PLAN.json"
        plan_path.write_text(json.dumps(frozen), encoding="utf-8")
        f.consumer.pins = MappingProxyType({**f.pins, "expected_game_plan_sha256": digest(plan_path)})
        f.consumer.source_scope = {"plan_path": str(plan_path), "batch_index": 0}
        claim_path = models.SECURITY500_GRANT_DIR / (f.pins["expected_window_sha256"] + ".CLAIM.json")
        claim_path.write_text(json.dumps({"window_sha256": f.pins["expected_window_sha256"],
            "game_plan_sha256": digest(plan_path), "destination": str(directory.resolve()),
            "initial_phase": "all"}), encoding="utf-8")
        f.plan_path, f.config, f.batch, f.claim_path = plan_path, config, batch, claim_path
        return f

    def test_real_helper_validates_ft_only_and_refetches_current_window(self):
        f = self.make_consumer()
        self.assertEqual(models.security500_planning_core_ref(), f.descriptor)
        for _ in range(2):
            self.assertIs(f.consumer.validate_request(f.handoff.FT)["metadata_request_allowed"], True)
        self.assertEqual(f.reads.count(f.index["request_window"]["path"]), 2)
        self.assertEqual(f.health.call_count, 2)
        with self.assertRaises(ValueError):
            f.consumer.validate_request(f.handoff.BASE)

    def test_stage1_window_rejects_base_or_more_than_thirteen_hours(self):
        f = self.make_consumer()
        self.assertTrue(f.handoff.validate_window(f.window))
        for field, value in (("allowed_model_ids", list(f.handoff.ARMS)),
                             ("expires_utc", "1970-01-01T13:01:01+00:00")):
            with self.subTest(field=field):
                invalid = copy.deepcopy(f.window)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    f.handoff.validate_window(invalid)

    def test_revoked_window_and_wrong_plan_stop_without_dispatch(self):
        f = self.make_consumer()
        with self.assertRaises(ValueError):
            f.consumer.require_plan("0" * 64)
        f = self.make_consumer()
        f.records[f.index["request_window"]["path"]] = b'{"status":"REVOKED"}'
        with self.assertRaises(ValueError):
            f.consumer.validate_request(f.handoff.FT)
        f.health.assert_not_called()
        f.records[f.index["request_window"]["path"]] = f.raw["window"]
        with self.assertRaises(ValueError):
            f.consumer.validate_request(f.handoff.FT)

    def test_expired_grant_and_stale_health_fail_closed(self):
        for mode in ("expiry", "health"):
            f = self.make_consumer()
            if mode == "expiry":
                f.clock.return_value = 180.0
            else:
                f.health.side_effect = lambda binding: dict(f.args["fresh_health"], updated_epoch=0.0)
            with self.assertRaises(ValueError):
                f.consumer.validate_request(f.handoff.FT)
            if mode == "expiry":
                f.health.assert_not_called()
            with self.assertRaises(ValueError):
                f.consumer.validate_request(f.handoff.FT)

    def test_persistent_stop_denies_before_reading_handoff_or_health(self):
        f = self.make_consumer()
        stopped = models.SECURITY500_GRANT_DIR / (f.pins["expected_window_sha256"] + ".STOP.json")
        stopped.write_text('{"status":"STOPPED"}', encoding="utf-8")
        f.reads.clear()
        with self.assertRaises(ValueError):
            f.consumer.validate_request(f.handoff.FT)
        self.assertEqual(f.reads, [])
        f.health.assert_not_called()
        stopped.unlink()
        with self.assertRaises(ValueError):
            f.consumer.validate_request(f.handoff.FT)

    def test_actual_http_hooks_validate_each_route_and_reject_changed_profile(self):
        for model_id in ("sp_lense_security500_neutral", "sp_lense_security500"):
            f = self.make_consumer()
            model, client = fake_model(model_id)
            self.mark_wire_fixture(f.consumer)
            f.consumer.install_http_guard(model, model_id)
            request = wire(model_id)
            self.assertEqual(asyncio.run(model.generate(request)), "synthetic dispatched response")
            self.assertEqual(f.health.call_count, 2)
            body = json.loads(request.content); body["temperature"] = 0.2
            request.content = json.dumps(body).encode()
            request.headers = {}
            from src.hosted_inspect_runner import IncompleteResponse
            with self.assertRaises(IncompleteResponse):
                asyncio.run(model.generate(request))
            # The entry check reads health, but malformed wire does not.
            self.assertEqual(f.health.call_count, 3)

    def test_cached_response_without_http_is_rejected(self):
        from src.hosted_inspect_runner import IncompleteResponse
        f = self.make_consumer()
        model, _ = fake_model("sp_lense_security500_neutral")
        self.mark_wire_fixture(f.consumer)
        async def cached(*args, **kwargs):
            return "synthetic replay without HTTP"
        model.generate = cached
        f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
        with self.assertRaises(IncompleteResponse):
            asyncio.run(model.generate([]))

    def test_nonzero_retries_cannot_dispatch_a_security500_client(self):
        from src.hosted_inspect_runner import IncompleteResponse
        f = self.make_consumer()
        model, client = fake_model("sp_lense_security500_neutral")
        self.mark_wire_fixture(f.consumer)
        model.config.max_retries = 1
        f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
        with self.assertRaises(IncompleteResponse):
            asyncio.run(model.generate(wire("sp_lense_security500_neutral")))
        f.health.assert_not_called()

    def test_sdk_retries_and_requested_response_cache_are_denied(self):
        from src.hosted_inspect_runner import IncompleteResponse
        for setting in ("sdk_retry", "cache", "positional_cache", "config_retry"):
            with self.subTest(setting=setting):
                f = self.make_consumer()
                model, client = fake_model("sp_lense_security500_neutral")
                self.mark_wire_fixture(f.consumer)
                if setting == "sdk_retry":
                    model.api.client.max_retries = 2
                f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
                kwargs = ({"cache": True} if setting == "cache" else
                          {"config": SimpleNamespace(max_retries=1)} if setting == "config_retry" else {})
                args = (wire("sp_lense_security500_neutral"), [], None, None, True) if setting == "positional_cache" else (wire("sp_lense_security500_neutral"),)
                with self.assertRaises(IncompleteResponse):
                    asyncio.run(model.generate(*args, **kwargs))
                f.health.assert_not_called()

    def test_sdk_client_or_hook_rebind_is_denied_before_dispatch(self):
        from src.hosted_inspect_runner import IncompleteResponse
        for setting in ("sdk_client", "http_hook", "reinit_options", "retry_policy"):
            with self.subTest(setting=setting):
                f = self.make_consumer()
                model, client = fake_model("sp_lense_security500_neutral")
                self.mark_wire_fixture(f.consumer)
                f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
                if setting == "sdk_client":
                    model.api.client = SimpleNamespace(_client=client, max_retries=0)
                elif setting == "http_hook":
                    client.event_hooks["request"].clear()
                elif setting == "reinit_options":
                    model.api.model_args["max_retries"] = 2
                else:
                    model.should_retry = lambda error: True
                with self.assertRaises(IncompleteResponse):
                    asyncio.run(model.generate(wire("sp_lense_security500_neutral")))
                f.health.assert_not_called()

    def test_hosted_comparators_use_zero_sdk_retries_and_no_response_cache(self):
        for model_id in ("terra", "deepseek"):
            with self.subTest(model_id=model_id):
                f = self.make_consumer()
                model, client = fake_model(model_id)
                f.consumer._child_plan_verified = True
                f.consumer._validate_game_source_scope = Mock(return_value={"models": [model_id]})
                model.config.max_retries = 3
                model.api.client.max_retries = 2
                model.api.model_args["max_retries"] = 2
                models._security500_guard_model(model, f.consumer, model_id)
                self.assertEqual(model.config.max_retries, 0)
                self.assertIs(model.config.cache, False)
                self.assertEqual(model.api.client.max_retries, 0)
                self.assertEqual(model.api.model_args["max_retries"], 0)
                self.assertIs(model.should_retry(RuntimeError("synthetic error")), False)
                self.assertEqual(asyncio.run(model.generate(SimpleNamespace(headers={}))), "synthetic dispatched response")
                f.health.assert_not_called()

    def test_second_http_attempt_cannot_return_completion(self):
        from src.hosted_inspect_runner import IncompleteResponse
        f = self.make_consumer()
        model, client = fake_model("sp_lense_security500_neutral")
        self.mark_wire_fixture(f.consumer)
        async def retried(request, *, cache=None):
            for attempt in range(2):
                for hook in client.event_hooks["request"]:
                    await hook(request)
            return "must never return"
        model.generate = retried
        f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
        with self.assertRaises(IncompleteResponse):
            asyncio.run(model.generate(wire("sp_lense_security500_neutral")))

    def test_valid_child_matches_real_synthetic_parent_and_requires_validation(self):
        f = self.source_scoped_fixture()
        model, _ = fake_model("sp_lense_security500_neutral")
        with self.assertRaises(ValueError):
            f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
        with patch("src.company_world_evaluation.source_matches", return_value=True):
            f.consumer.validate_child_plan(copy.deepcopy(f.batch))
            self.assertTrue(f.consumer._child_plan_verified)
            self.assertEqual(f.consumer._validate_game_source_scope(), f.batch)

    def test_mismatched_child_plan_stops_before_http(self):
        f = self.source_scoped_fixture()
        child = copy.deepcopy(f.batch); child["models"] = ["sp_lense_security500"]
        with patch("src.company_world_evaluation.source_matches", return_value=True):
            with self.assertRaises(ValueError):
                f.consumer.validate_child_plan(child)
        model, client = fake_model("sp_lense_security500_neutral")
        with self.assertRaises(ValueError):
            f.consumer.install_http_guard(model, "sp_lense_security500_neutral")
        self.assertEqual(client.event_hooks["request"], [])
        f.health.assert_not_called()

    def test_changed_parent_config_source_or_batch_route_denies_before_dispatch(self):
        from src.hosted_inspect_runner import IncompleteResponse
        for failure in ("parent", "config", "source", "route", "claim_path", "claim_digest", "stop"):
            with self.subTest(failure=failure):
                f = self.source_scoped_fixture()
                model_id = "sp_lense_security500" if failure == "route" else "sp_lense_security500_neutral"
                model, _ = fake_model(model_id)
                # Metadata authorization is covered by the real-helper tests;
                # isolate the actual parent-plan check before the HTTP attempt.
                f.consumer.validate_request = Mock(return_value={"metadata_request_allowed": True})
                with patch("src.company_world_evaluation.source_matches", return_value=True) as sources:
                    f.consumer.validate_child_plan(copy.deepcopy(f.batch))
                    f.consumer.install_http_guard(model, model_id)
                    if failure == "parent":
                        f.plan_path.write_text("{}", encoding="utf-8")
                    elif failure == "config":
                        f.config.write_text("changed: true\n", encoding="utf-8")
                    elif failure == "source":
                        sources.return_value = False
                    elif failure == "claim_path":
                        claim = json.loads(f.claim_path.read_text(encoding="utf-8"))
                        claim["destination"] = str(f.plan_path.parent / "wrong")
                        f.claim_path.write_text(json.dumps(claim), encoding="utf-8")
                    elif failure == "claim_digest":
                        claim = json.loads(f.claim_path.read_text(encoding="utf-8"))
                        claim["game_plan_sha256"] = "0" * 64
                        f.claim_path.write_text(json.dumps(claim), encoding="utf-8")
                    elif failure == "stop":
                        stopped = models.SECURITY500_GRANT_DIR / (f.pins["expected_window_sha256"] + ".STOP.json")
                        stopped.write_text('{"status":"STOPPED"}', encoding="utf-8")
                    with self.assertRaises(IncompleteResponse):
                        asyncio.run(model.generate(wire(model_id)))
                f.health.assert_not_called()
