"""Checkpoint isolation and matched comparison preparation without model calls."""
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src import company_models as models
from src import infrastructure_preservation_comparison as runner
from src.company_preservation_comparison import summarize
from src.hosted_inspect_runner import hosted_model_config, reasoning_kind_for_model


NEW_WEIGHT = "1" * 64
NEW_CONDITIONS = ("sp_lense_retrained_neutral", "sp_lense_retrained")
TEMPLATE = runner.REPOSITORY / "config/curriculum_revision_48h_template.yaml"


@contextmanager
def frozen_test_checkpoint(weight=NEW_WEIGHT, config=models.LAST8_CONFIG_SHA):
    with patch.dict(models.MODEL_SPECS[NEW_CONDITIONS[0]], adapter_sha256=weight, adapter_config_sha256=config), \
            patch.dict(models.MODEL_SPECS[NEW_CONDITIONS[1]], adapter_sha256=weight, adapter_config_sha256=config):
        yield


def qualified_binding(root, *, comparison=True):
    routes = ["qwen3.8-27b-base", models.LAST8_SERVICE_MODEL_ID]
    limits = {"context_total_tokens": 131072, "max_context_tokens_including_output": 131072,
              "max_output_tokens": 8192}
    expected = {"ready": True, "base_revision": models.LOCAL_BASE_REVISION,
                "base_manifest_sha256": models.LOCAL_BASE_MANIFEST_SHA256,
                "model_ids": routes, "max_concurrency": 1, "resident_base_count": 1,
                "limits": limits, "last_post_request_check": "passed", "pid": 123,
                "process_created_at": 1.0, "source_sha256": "2" * 64, "manifest_sha256": "3" * 64}
    receipt_data = {key: expected[key] for key in ("source_sha256", "manifest_sha256", "pid", "process_created_at", "limits")}
    receipt_data.update(base_revision=models.LOCAL_BASE_REVISION,
                        base_manifest_sha256=models.LOCAL_BASE_MANIFEST_SHA256, model_ids=routes)
    binding = {"schema": "research4.game-last8-service-binding.v1", "adapter_sha256": models.LAST8_SHA,
               "base_url": "http://127.0.0.1:8769/v1", "health_url": "http://127.0.0.1:8769/health",
               "expected_health": expected}
    if comparison:
        routes.append(models.RETRAINED_SERVICE_MODEL_ID)
        binding["schema"] = models.COMPARISON_BINDING_SCHEMA
        binding["per_model"] = {
            routes[0]: {"adapter_enabled": False},
            routes[1]: {"adapter_enabled": True, "adapter_sha256": models.LAST8_SHA,
                        "adapter_config_sha256": models.LAST8_CONFIG_SHA, "rank": 16,
                        "blocks": list(range(56, 64)), "projections": 62},
            routes[2]: models.retrained_identity(),
        }
        expected["per_model"] = copy.deepcopy(binding["per_model"])
        receipt_data["per_model"] = copy.deepcopy(binding["per_model"])
    else:
        expected.update(adapter_sha256=models.LAST8_SHA, adapter_config_sha256=models.LAST8_CONFIG_SHA)
    receipt = root / "qualified_root_handoff.json"
    receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    binding.update(readiness_receipt_path=str(receipt),
                   readiness_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    path = root / "game_binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    return path, binding


def unit_config(path):
    config = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    config.update(role="unit_test", seeds=[101, 102, 103])
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


class ModelMatrixPreparationTests(unittest.TestCase):
    LEGACY = ("qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8")
    COMPARISON = LEGACY + NEW_CONDITIONS
    PROVIDER = COMPARISON + ("sp_lense_provider352_neutral", "sp_lense_provider352")
    HOSTED = ("terra", "deepseek")

    def test_exact_four_six_and_eight_local_metadata_preserves_both_input_orders(self):
        with patch.object(runner, "selected_models", side_effect=AssertionError("Metadata is not admission")), \
                patch.object(runner, "prospective_plan", side_effect=AssertionError("No case planning")), \
                patch.object(Path, "read_text", side_effect=AssertionError("No metadata files")):
            for local in (self.LEGACY, self.COMPARISON, self.PROVIDER):
                for ordered_local, ordered_hosted in ((local, self.HOSTED),
                                                     (tuple(reversed(local)), tuple(reversed(self.HOSTED)))):
                    with self.subTest(local_count=len(local), hosted=ordered_hosted):
                        self.assertEqual(runner.validate_model_matrix(ordered_local, ordered_hosted),
                                         (ordered_local, ordered_hosted))
            local, hosted = runner.validate_model_matrix(self.PROVIDER, self.HOSTED)
            self.assertEqual(len(local) + len(hosted), 10)

    def test_partial_extra_substituted_and_duplicate_matrices_are_rejected(self):
        invalid_local = (
            self.LEGACY + (self.LEGACY[0],),
            self.COMPARISON + ("sp_lense_provider352_neutral",),
            self.LEGACY + ("sp_lense_provider352_neutral", "sp_lense_provider352"),
            self.PROVIDER + ("qwen_max",),
            ("sp_lense_neutral",) + self.LEGACY[1:],
        )
        for local in invalid_local:
            with self.subTest(local=local), self.assertRaisesRegex(ValueError, "local conditions"):
                runner.validate_model_matrix(local, self.HOSTED)
        for hosted in (("terra",), ("terra", "terra"), ("terra", "qwen_max"),
                       self.HOSTED + ("qwen_max",)):
            with self.subTest(hosted=hosted), self.assertRaisesRegex(ValueError, "hosted references"):
                runner.validate_model_matrix(self.PROVIDER, hosted)

    def test_provider_remains_not_ready_before_planning_writes_or_dispatch(self):
        # Minimal in-memory fixture only; never open a study configuration or cases.
        fixture = {"seeds": (1, 2, 3), "local_models": self.PROVIDER,
                   "hosted_models": self.HOSTED, "split_hosted_batches": True}
        with patch.object(Path, "read_text", return_value="unit fixture"), \
                patch.object(runner.yaml, "safe_load", return_value=fixture), \
                patch.object(runner, "prospective_plan") as prospective, \
                patch.object(runner, "write_json") as write, \
                patch.object(runner, "request_window") as window, \
                patch.object(runner, "current_local_binding") as binding, \
                patch.object(runner, "run_batch") as dispatch, \
                patch.object(models, "build_model") as build, \
                patch.object(models.urllib.request, "urlopen") as health:
            for count in (None, 180, 2160):
                placeholders = {"completed_training_updates": count, "provider_DEV_rows": 500,
                                "adapter_sha256": "a" * 64, "adapter_config_sha256": "b" * 64,
                                "provider_readiness": "READY", "execution_enabled": True}
                with patch.dict(models.MODEL_SPECS["sp_lense_provider352_neutral"], placeholders), \
                        patch.dict(models.MODEL_SPECS["sp_lense_provider352"], placeholders):
                    with self.subTest(count=count, operation="plan"), self.assertRaisesRegex(ValueError, "NOT_READY"):
                        runner.plan(Path("unused-fixture.yaml"))
                    with self.subTest(count=count, operation="execute"), self.assertRaisesRegex(ValueError, "NOT_READY"):
                        runner.execute_locked(Path("unused-output"), Path("unused.env"),
                                              Path("unused-fixture.yaml"), "all")
            prospective.assert_not_called()
            write.assert_not_called()
            window.assert_not_called()
            binding.assert_not_called()
            dispatch.assert_not_called()
            build.assert_not_called()
            health.assert_not_called()


class RequestWindowHandoffTests(unittest.TestCase):
    def test_legacy_and_compact_grants_require_their_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window, lock, binding = root / 'window.json', root / 'claim.lock', root / 'binding.json'
            destination = root / 'run'; destination.mkdir()
            binding.write_text('{"qualified":"binding"}', encoding='utf-8')
            binding_sha = hashlib.sha256(binding.read_bytes()).hexdigest()
            health = {'pid':77, 'process_created_at':123.0}
            for status, digest in [('GAME_128K_ENDPOINT_HANDED_OFF','receipt-hash'),
                                   ('GAME_REQUEST_WINDOW_GRANTED_AFTER_AUDITED_COMPACT_V2_FIT',binding_sha)]:
                grant = {'status':status,'owner_thread':runner.OWNER,'Game_requests_allowed':True,
                         'service_pid':77,'service_process_created_at':123.0,'handoff_sha256':digest}
                window.write_text(json.dumps(grant), encoding='utf-8')
                with patch.object(runner,'WINDOW',window), patch.object(runner,'LOCK',lock), \
                        patch.dict(runner.os.environ,RESEARCH4_LAST8_SERVICE_BINDING=str(binding)), \
                        patch.object(runner,'current_local_binding',return_value={'readiness_receipt_sha256':'receipt-hash'}), \
                        patch.object(runner,'assert_local_binding',return_value=health), \
                        patch.object(runner,'idle_service',return_value=health):
                    with runner.request_window(destination,'frozen-plan'):
                        self.assertEqual(json.loads(window.read_text())['status'],'ACTIVE_GAME_COLLECTION')
                    self.assertEqual(json.loads(window.read_text())['status'],'AVAILABLE_NO_ACTIVE_COLLECTION')
                    self.assertFalse(lock.exists())
                grant['handoff_sha256']='wrong'
                window.write_text(json.dumps(grant), encoding='utf-8')
                with patch.object(runner,'WINDOW',window), patch.object(runner,'LOCK',lock), \
                        patch.dict(runner.os.environ,RESEARCH4_LAST8_SERVICE_BINDING=str(binding)), \
                        patch.object(runner,'current_local_binding',return_value={'readiness_receipt_sha256':'receipt-hash'}), \
                        patch.object(runner,'assert_local_binding') as probe:
                    with self.assertRaisesRegex(ValueError,'differs from'):
                        with runner.request_window(destination,'frozen-plan'):
                            self.fail('Mismatched grant must never admit work')
                    probe.assert_not_called()
                    self.assertFalse(lock.exists())


class RetrainedBindingTests(unittest.TestCase):
    def test_project_relative_receipt_uses_binding_project_not_current_checkout(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / ".local/serving/qualified"
            folder.mkdir(parents=True)
            path, binding = qualified_binding(folder)
            binding["readiness_receipt_path"] = ".local/serving/qualified/qualified_root_handoff.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            self.assertEqual(models.current_local_binding(path, required_models=list(NEW_CONDITIONS)), binding)
            binding["readiness_receipt_path"] = "../qualified_root_handoff.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay inside"):
                models.current_local_binding(path, required_models=list(NEW_CONDITIONS))

    def test_pending_checkpoint_fails_before_binding_or_provider(self):
        with patch.dict(models.MODEL_SPECS[NEW_CONDITIONS[0]], adapter_sha256=None, adapter_config_sha256=None), \
                patch.dict(models.MODEL_SPECS[NEW_CONDITIONS[1]], adapter_sha256=None, adapter_config_sha256=None), \
                patch.object(models, "current_local_binding") as binding, \
                patch.object(models, "build_model") as build, \
                patch.object(models.urllib.request, "urlopen") as health:
            for condition in NEW_CONDITIONS:
                with self.assertRaisesRegex(ValueError, "qualification is pending"):
                    models.build_company_model(condition)
            binding.assert_not_called()
            build.assert_not_called()
            health.assert_not_called()

    def test_old_weights_cannot_be_new_but_same_geometry_config_is_valid(self):
        with frozen_test_checkpoint(weight=models.LAST8_SHA):
            with self.assertRaisesRegex(ValueError, "old checkpoint"):
                models.selected_models([NEW_CONDITIONS[0]])
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, binding = qualified_binding(Path(temporary))
            with patch.object(models.urllib.request, "urlopen") as health:
                self.assertEqual(models.current_local_binding(path, required_models=list(NEW_CONDITIONS)), binding)
                health.assert_not_called()

    def test_legacy_binding_still_works_and_cannot_supply_new_route(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, binding = qualified_binding(Path(temporary), comparison=False)
            self.assertEqual(models.current_local_binding(path, required_models=["qwen_base"]), binding)
            with patch.object(models.urllib.request, "urlopen") as health:
                with self.assertRaisesRegex(ValueError, "selected route"):
                    models.current_local_binding(path, required_models=[NEW_CONDITIONS[0]])
                health.assert_not_called()

    def test_route_proof_mutations_reject_before_any_provider_or_health_call(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, original = qualified_binding(Path(temporary))
            mutations = [
                lambda x: x["per_model"][models.RETRAINED_SERVICE_MODEL_ID].update(adapter_sha256=models.LAST8_SHA),
                lambda x: x["per_model"][models.LAST8_SERVICE_MODEL_ID].update(adapter_sha256=NEW_WEIGHT),
                lambda x: x["per_model"][models.RETRAINED_SERVICE_MODEL_ID].update(rank=32),
                lambda x: x["expected_health"]["model_ids"].remove(models.RETRAINED_SERVICE_MODEL_ID),
                lambda x: x["expected_health"].update(source_sha256="4" * 64),
                lambda x: x["expected_health"].update(last_post_request_check="failed"),
                lambda x: x["expected_health"].update(max_concurrency=2),
                lambda x: x["expected_health"]["limits"].update(max_output_tokens=16384),
                lambda x: x.update(readiness_receipt_sha256="0" * 64),
            ]
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(path)}), \
                    patch.object(models, "build_model") as build, \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models.urllib.request, "urlopen") as health:
                for mutate in mutations:
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        models.build_company_model(NEW_CONDITIONS[0])
                build.assert_not_called()
                health.assert_not_called()

    def test_new_route_uses_identical_native_sampling_and_builds_only_its_id(self):
        self.assertEqual(hosted_model_config(models.RETRAINED_MODEL),
                         {**hosted_model_config(models.LAST8_MODEL), "model": models.RETRAINED_MODEL})
        self.assertEqual(reasoning_kind_for_model(models.RETRAINED_MODEL), "raw")
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, _ = qualified_binding(Path(temporary))
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(path),
                                               "RESEARCH4_QWEN_BASE_URL": models.QWEN_BASE_URL}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models, "build_model") as build:
                models.build_company_model(NEW_CONDITIONS[0])
                build.assert_called_once_with(models.RETRAINED_MODEL, 1, 8192)

    def test_matching_placeholder_source_fields_do_not_establish_a_qualified_service(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, binding = qualified_binding(Path(temporary))
            receipt = Path(binding["readiness_receipt_path"])
            qualified = json.loads(receipt.read_text(encoding="utf-8"))
            qualified["source_sha256"] = None
            binding["expected_health"]["source_sha256"] = None
            receipt.write_text(json.dumps(qualified), encoding="utf-8")
            binding["readiness_receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            path.write_text(json.dumps(binding), encoding="utf-8")
            with patch.object(models.urllib.request, "urlopen") as health:
                with self.assertRaisesRegex(ValueError, "source proof"):
                    models.current_local_binding(path)
                health.assert_not_called()

    def test_expected_health_must_bind_the_same_route_map_before_a_probe(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, original = qualified_binding(Path(temporary))
            missing = copy.deepcopy(original)
            del missing["expected_health"]["per_model"]
            swapped = copy.deepcopy(original)
            routes = swapped["expected_health"]["per_model"]
            routes[models.LAST8_SERVICE_MODEL_ID]["adapter_sha256"] = NEW_WEIGHT
            routes[models.RETRAINED_SERVICE_MODEL_ID]["adapter_sha256"] = models.LAST8_SHA
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(path)}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models, "build_model") as build, \
                    patch.object(models.urllib.request, "urlopen") as health:
                for changed in (missing, swapped):
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Expected health"):
                        models.build_company_model(NEW_CONDITIONS[0])
                build.assert_not_called()
                health.assert_not_called()

    def test_fresh_health_rejects_missing_or_swapped_loaded_route_identities(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path, binding = qualified_binding(Path(temporary))
            correct = copy.deepcopy(binding["expected_health"])
            missing = copy.deepcopy(correct)
            del missing["per_model"]
            swapped = copy.deepcopy(correct)
            swapped["per_model"][models.LAST8_SERVICE_MODEL_ID]["adapter_sha256"] = NEW_WEIGHT
            swapped["per_model"][models.RETRAINED_SERVICE_MODEL_ID]["adapter_sha256"] = models.LAST8_SHA
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(path),
                                               "RESEARCH4_QWEN_BASE_URL": binding["base_url"]}), \
                    patch.object(models, "local_reservation", return_value=None), \
                    patch.object(models, "build_model") as build:
                for actual in (missing, swapped):
                    response = io.BytesIO(json.dumps(actual).encode())
                    with patch.object(models.urllib.request, "urlopen", return_value=response) as health:
                        with self.assertRaisesRegex(ValueError, "does not match"):
                            models.assert_local_binding(NEW_CONDITIONS[0])
                        health.assert_called_once_with(binding["health_url"], timeout=5)
                with patch.object(models.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(correct).encode())):
                    self.assertEqual(models.assert_local_binding(NEW_CONDITIONS[0]), correct)
                build.assert_not_called()


class CurriculumComparisonTests(unittest.TestCase):
    def test_public_template_has_no_usable_seeds(self):
        template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(template["seeds"], [])
        with self.assertRaisesRegex(ValueError, "prospective seeds"):
            runner.plan(TEMPLATE)

    def test_fake_worlds_form_twenty_four_unique_matched_cases(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit.yaml"
            config = unit_config(path)
            frozen = runner.plan(path)
        self.assertEqual(frozen["episodes"], 24)
        self.assertEqual(len(frozen["batches"]), 18)
        self.assertEqual(frozen["comparison_contrasts"], config["comparison_contrasts"])
        self.assertEqual(frozen["primary_contrast"], ["sp_lense_retrained_neutral", "sp_lense_last8_neutral"])
        self.assertEqual(frozen["acceptance_criterion"], config["acceptance_criterion"])
        self.assertEqual(frozen["acceptance_criterion"]["minimum_successes"], 2)
        self.assertEqual(frozen["acceptance_criterion"]["total_episodes"], 3)
        self.assertTrue(frozen["acceptance_criterion"]["strictly_more_successes_than_comparator"])
        self.assertIn(["sp_lense_retrained_neutral", "qwen_base"], frozen["comparison_contrasts"])
        self.assertIn(["sp_lense_last8_neutral", "qwen_base"], frozen["comparison_contrasts"])
        samples = [s for b in frozen["batches"] for s in b["samples"]]
        self.assertEqual(len({s["case_id"] for s in samples}), 24)
        for seed in (101, 102, 103):
            batches = [b for b in frozen["batches"] if b["seed"] == seed]
            self.assertEqual(len(batches), 6)
            self.assertEqual(len({b["world_plan_sha256"] for b in batches}), 1)
            self.assertEqual(len({b["initial_user_payload_sha256"] for b in batches}), 1)
            self.assertEqual(batches[0]["models"][1:], ["terra", "deepseek"])
            specs = {s["model_id"]: s for b in batches for s in b["samples"]}
            for left, right in config["same_prompt_adapter_contrasts"]:
                self.assertEqual(specs[left]["system_prompt_sha256"], specs[right]["system_prompt_sha256"])
            for batch in batches:
                self.assertEqual(sum(bool(models.MODEL_SPECS[m].get("local_qwen")) for m in batch["models"]), 1)
                self.assertEqual((batch["horizon_hours"], batch["outer_seconds"]), (48, 14400))
                self.assertTrue(batch["thinking_enabled"])
                self.assertEqual(batch["reasoning_effort"], "low")

    def test_missing_new_route_blocks_whole_collection_before_any_provider(self):
        with frozen_test_checkpoint(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "unit.yaml"
            unit_config(config_path)
            binding_path, _ = qualified_binding(root, comparison=False)
            output = root / "output"
            output.mkdir()
            with patch.dict(models.os.environ, {"RESEARCH4_LAST8_SERVICE_BINDING": str(binding_path)}), \
                    patch.object(runner, "run_batch") as submit, \
                    patch.object(models, "build_model") as build, \
                    patch.object(models.urllib.request, "urlopen") as health:
                with self.assertRaisesRegex(ValueError, "selected route"):
                    runner.execute_locked(output, Path("unused.env"), config_path, "all")
                submit.assert_not_called()
                build.assert_not_called()
                health.assert_not_called()
                self.assertFalse((output / "PLAN.json").exists())

    def test_summary_reports_configured_new_old_and_base_contrasts(self):
        cells = [{"model_id": model, "seed": 101, "survived_horizon": model == "sp_lense_retrained",
                  "evidence_status": "audited_complete"}
                 for model in ("qwen_base", "sp_lense_last8", "sp_lense_retrained")]
        pairs = [["sp_lense_retrained", "qwen_base"], ["sp_lense_retrained", "sp_lense_last8"]]
        summary = summarize(cells, models=[r["model_id"] for r in cells], seeds=[101], contrasts=pairs)
        self.assertEqual([(r["left"], r["right"]) for r in summary["paired_survival_comparisons"]],
                         [tuple(pair) for pair in pairs])
        self.assertTrue(all(r["win"] == 1 for r in summary["paired_survival_comparisons"]))
        with self.assertRaises(ValueError):
            summarize(cells, models=[r["model_id"] for r in cells], seeds=[101], contrasts=[["missing", "qwen_base"]])


if __name__ == "__main__":
    unittest.main()
