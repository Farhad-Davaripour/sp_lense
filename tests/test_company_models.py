"""Choice visibility and exact provider routing; never call an inference API."""
from pathlib import Path
import hashlib
import json
import copy
import io
import tempfile
from unittest import TestCase
from unittest.mock import patch

from src import company_models as models
from src.hosted_inspect_runner import hosted_model_config

QWEN_ENV = {"QWEN_API_KEY": "fixture", "QWEN_BASE_URL": models.qwen_cloud.BASE_URL,
            "QWEN_MODEL": "qwen3.8-max"}
CURRENT_LOCAL_MODELS = ("qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8")


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
            "limits": {"max_output_tokens": 8192, "context_total_tokens": 32768,
                       "max_context_tokens_including_output": 32768}}}
    path = root / "new_binding.json"
    path.write_text(json.dumps(binding))
    return path, binding


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
            self.assertEqual(len(choices), 8)
            self.assertEqual({m["id"] for m in choices if m["available"]},
                             {key for key, spec in models.MODEL_SPECS.items() if not spec.get("retired")})
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
                if spec.get("retired"):
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
                self.assertEqual(len(held), 8)
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
