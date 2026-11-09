"""Choice visibility and exact provider routing; never call an inference API."""
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from src import company_models as models
from src.hosted_inspect_runner import hosted_model_config


class CompanyModelTests(TestCase):
    def test_full_catalog_does_not_shrink_to_the_operator_test_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / ".env"
            credentials.write_text("OPENAI_API_KEY=fixture\nDEEPSEEK_API_KEY=fixture\n")
            with patch.object(models, "local_binding", return_value={}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models.urllib.request, "urlopen", side_effect=AssertionError("No readiness probe for catalog")):
                choices = models.model_choices(credentials)
            self.assertEqual(len(choices), 7)
            self.assertEqual({m["id"] for m in choices if m["available"]}, set(models.MODEL_SPECS))
            self.assertEqual([m["id"] for m in choices if m["default_selected"]], ["deepseek"])
            self.assertFalse(choices[-1]["available"])
            self.assertIn("not verified", choices[-1]["status"])

    def test_neutral_and_instruction_arms_bind_the_same_exact_weights(self):
        for neutral, instructed in (("qwen_base_neutral", "qwen_base"), ("sp_lense_neutral", "sp_lense")):
            self.assertEqual(models.MODEL_SPECS[neutral]["model"], models.MODEL_SPECS[instructed]["model"])
            self.assertEqual(models.MODEL_SPECS[neutral]["prompt_condition"], "baseline")
            self.assertEqual(models.MODEL_SPECS[instructed]["prompt_condition"], "security_maximization")
        self.assertTrue(models.MODEL_SPECS["sp_lense"]["model"].endswith("sp-lense-adaptor-last-layer"))
        self.assertNotIn("shared-adapter", models.MODEL_SPECS["sp_lense"]["model"])

    def test_shared_builder_uses_each_models_supported_output_budget(self):
        with patch.object(models, "build_model") as build, patch.object(models, "local_reservation", return_value=None), patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": models.QWEN_BASE_URL}):
            for name, spec in models.MODEL_SPECS.items():
                models.build_company_model(name)
                build.assert_called_with(spec["model"], 1, 4096 if spec.get("local_qwen") else 8192)
        config = hosted_model_config(models.LAST_MODEL)
        self.assertTrue(config["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_stale_local_client_url_is_rejected_before_any_model_build(self):
        with patch.dict(models.os.environ, {"RESEARCH4_QWEN_BASE_URL": "http://127.0.0.1:9999/v1"}), \
                patch.object(models, "local_reservation", return_value=None), \
                patch.object(models, "build_model") as build:
            with self.assertRaisesRegex(ValueError, "reviewed Mac endpoint"):
                models.build_company_model("qwen_base")
            build.assert_not_called()

    def test_reservation_keeps_local_choices_visible_and_releases_live(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "reservation.json"
            credentials = Path(directory) / ".env"
            credentials.write_text("OPENAI_API_KEY=fixture\nDEEPSEEK_API_KEY=fixture\n")
            with patch.dict(models.os.environ, {"RESEARCH4_QWEN_RESERVATION_FILE": str(marker)}), \
                    patch.object(models, "local_binding", return_value={}):
                marker.write_text('{"active":true,"purpose":"paired evaluation"}')
                held = models.model_choices(credentials)
                self.assertEqual(len(held), 7)
                self.assertEqual({c["id"] for c in held if c["available"]}, {"terra", "deepseek"})
                for choice in held[:4]:
                    self.assertIn("Temporarily reserved", choice["status"])
                with patch.object(models.urllib.request, "urlopen") as health, patch.object(models, "build_model") as build:
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        models.assert_local_binding()
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        models.build_company_model("sp_lense")
                    health.assert_not_called()
                    build.assert_not_called()
                marker.write_text('{"active":false}')
                released = models.model_choices(credentials)
                self.assertEqual(sum(c["available"] for c in released), 6)

    def test_unreadable_reservation_only_holds_local_models(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "reservation.json"
            marker.write_text("partial json")
            with patch.dict(models.os.environ, {"RESEARCH4_QWEN_RESERVATION_FILE": str(marker)}):
                self.assertIsNotNone(models.local_reservation())
