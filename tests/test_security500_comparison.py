"""CPU-only security500 comparison handoff checks; no private study data."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import infrastructure_preservation_comparison as runner


class Security500RunnerTests(unittest.TestCase):
    def setUp(self):
        self.grants = tempfile.TemporaryDirectory()
        self.addCleanup(self.grants.cleanup)
        patcher = patch.object(runner, "SECURITY500_GRANT_DIR", Path(self.grants.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def consumer(self, plan_hash="f" * 64):
        core = {"core_binding": {"path": ".local/unit/core.json", "sha256": "a" * 64},
                "trusted_review_sha256": "b" * 64}
        result = Mock()
        result.core_binding = core
        result.pins = {"expected_game_plan_sha256": plan_hash, "expected_window_sha256": "c" * 64}
        result.contract = SimpleNamespace(ARMS=("base", "security500"))
        result.private_child_context.return_value = "unit-private-scope"
        return result

    def test_exact_new_matrix_keeps_old_matrices_and_rejects_mixtures(self):
        hosted = ("terra", "deepseek")
        local = ("qwen_base_neutral", "qwen_base", "sp_lense_security500_neutral", "sp_lense_security500")
        self.assertEqual(runner.validate_model_matrix(local, hosted), (local, hosted))
        legacy = ("qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8")
        self.assertEqual(runner.validate_model_matrix(legacy, hosted), (legacy, hosted))
        with self.assertRaises(ValueError):
            runner.validate_model_matrix((*local[:-1], "sp_lense_last8"), hosted)
        with self.assertRaises(ValueError):
            runner.validate_model_matrix((*local, local[0]), hosted)

    def test_screen_matrix_accepts_only_ft_neutral_without_hosted(self):
        local = ("sp_lense_security500_neutral",)
        self.assertEqual(runner.validate_model_matrix(local, ()), (local, ()))
        for bad_local, bad_hosted in ((local, ("terra",)),
                                      (("sp_lense_security500",), ()),
                                      (("sp_lense_security500_neutral", "qwen_base"), ())):
            with self.subTest(local=bad_local, hosted=bad_hosted), self.assertRaises(ValueError):
                runner.validate_model_matrix(bad_local, bad_hosted)

    def test_screen_plan_has_three_ft_episodes_without_empty_hosted_batches(self):
        config = {"schema": "screen", "role": "screen", "seeds": [101, 102, 103],
                  "local_models": ["sp_lense_security500_neutral"], "hosted_models": [],
                  "split_hosted_batches": True, "world_version": "company_world_long_horizon_v1", "horizon_hours": 48,
                  "outer_seconds": 14400, "primary_outcome": "completion", "primary_contrast": None,
                  "same_prompt_adapter_contrasts": [], "comparison_contrasts": []}
        def prospective(seed, *, models, **kwargs):
            self.assertEqual(models, ["sp_lense_security500_neutral"])
            return {"seed": seed, "models": models, "samples": [{"case_id": str(seed)}],
                    "horizon_hours": 48, "outer_seconds": 14400, "max_decisions": 256}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "selected_models"), patch.object(runner, "prospective_plan", side_effect=prospective) as build:
                result = runner.plan(path)
                self.assertEqual(result["episodes"], 3)
                self.assertEqual(result["models"], ("sp_lense_security500_neutral",))
                self.assertEqual(result["comparison_contrasts"], [])
                self.assertEqual(build.call_count, 3)
                config["seeds"] = list(range(9))
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly three"):
                    runner.plan(path)
                config["seeds"] = [101, 102, 103]
                config["horizon_hours"] = 24
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "committed 48-hour"):
                    runner.plan(path)
                config["horizon_hours"] = 48
                for field, value in (("primary_contrast", ["a", "b"]),
                                     ("same_prompt_adapter_contrasts", [["a", "b"]]),
                                     ("comparison_contrasts", [["a", "b"]])):
                    original = config[field]
                    config[field] = value
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cannot specify model comparisons"):
                        runner.plan(path)
                    config[field] = original
                path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(runner, "prospective_plan", side_effect=lambda seed, **kw:
                                  {**prospective(seed, **kw), "max_decisions": 255}):
                    with self.assertRaisesRegex(ValueError, "committed decision limit"):
                        runner.plan(path)

    def test_screen_never_reads_credentials_or_requests_base(self):
        ft = runner.MODEL_SPECS["sp_lense_security500_neutral"]["service_model_id"]
        candidate = {"models": ("sp_lense_security500_neutral",), "episodes": 3,
                     "batches": [{"models": ["sp_lense_security500_neutral"], "seed": i} for i in (101, 102, 103)],
                     "seeds": [101, 102, 103], "protocol": "unit", "horizon_hours": 48,
                     "comparison_contrasts": []}
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            runner.write_json(destination / "PLAN.json", candidate)
            def fake_batch(batch, directory, credentials, **kwargs):
                self.assertIsNone(credentials)
                return ([{"model_id": "sp_lense_security500_neutral", "seed": batch["seed"],
                          "evidence_status": "audited_complete"}],
                        {"batch": directory.name, "exit_code": 0, "failure": None})
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "provider352_planning_core_ref", return_value=None), \
                    patch.object(runner, "security500_planning_core_ref", return_value=consumer.core_binding), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "LOCK", destination / "claim.lock"), \
                    patch.object(runner, "run_batch", side_effect=fake_batch) as dispatch:
                runner.execute_locked(destination, Path("unused"), Path("unused"), "all", security500_consumer=consumer)
            self.assertEqual(dispatch.call_count, 3)
            self.assertTrue(consumer.validate_request.call_args_list)
            self.assertEqual({call.args[0] for call in consumer.validate_request.call_args_list}, {ft})

        with tempfile.TemporaryDirectory(dir=runner.REPOSITORY / ".local") as tmp:
            config_path = Path(tmp) / "screen.json"
            config_path.write_text(json.dumps({"local_models": ["sp_lense_security500_neutral"], "hosted_models": []}), encoding="utf-8")
            with patch.object(runner, "configured_key_names", side_effect=AssertionError("credential read")), \
                    patch.object(runner, "execute_locked") as execute:
                runner._execute_in_destination(Path(tmp) / "out", Path("unused"), config_path, "all", security500_consumer=consumer)
                execute.assert_called_once()

    def test_child_local_readback_uses_ft_for_screen_and_legacy_group_for_old_mode(self):
        from src import company_world_evaluation as evaluation
        with patch.dict(evaluation.os.environ, {evaluation.SECURITY500_CONTEXT_ENV: "synthetic"}), \
                patch.object(evaluation, "assert_local_binding", return_value={"ready": True}) as health:
            self.assertEqual(evaluation.local_binding_readback({"models": ["sp_lense_security500_neutral"]}),
                             {"last8": {"ready": True}})
            health.assert_called_once_with("sp_lense_security500_neutral")
        with patch.dict(evaluation.os.environ, {}, clear=True), \
                patch.object(evaluation, "assert_local_binding", return_value={"ready": True}) as health:
            self.assertEqual(evaluation.local_binding_readback({"models": ["qwen_base"]}),
                             {"last8": {"ready": True}})
            health.assert_called_once_with("last8")

    def test_security500_window_only_claims_locally_and_never_releases_root(self):
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            with patch.object(runner, "LOCK", destination / "claim.lock"), \
                    patch.object(runner, "update_window", side_effect=AssertionError("Root grant is read only")):
                with runner.security500_request_window(destination, "f" * 64, consumer, route="base"):
                    self.assertTrue((destination / "claim.lock").exists())
                self.assertFalse((destination / "claim.lock").exists())
            consumer.require_plan.assert_called_once_with("f" * 64)
            consumer.validate_request.assert_called_once_with("base")
            closeout = json.loads((destination / "SECURITY500_CONSUMER_CLOSEOUT.json").read_text())
            self.assertFalse(closeout["request_window_released"])
            self.assertFalse(closeout["root_window_release_confirmed"])

    def test_grant_rejects_other_destination_and_stopped_grant_reuse(self):
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            first.mkdir(); second.mkdir()
            runner.claim_security500_grant(first, "f" * 64, "hosted", consumer)
            with self.assertRaisesRegex(ValueError, "already claimed"):
                runner.claim_security500_grant(second, "f" * 64, "local", consumer)
            runner.stop_security500_collection(first, consumer)
            for directory in (first, second):
                with self.assertRaisesRegex(ValueError, "Stopped"):
                    runner.claim_security500_grant(directory, "f" * 64, "local", consumer)

    def test_only_finished_explicit_staged_phase_can_continue_once(self):
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            runner.claim_security500_grant(destination, "f" * 64, "hosted", consumer)
            with self.assertRaisesRegex(ValueError, "prior completed phase"):
                runner.claim_security500_grant(destination, "f" * 64, "local", consumer)
            runner.write_json(destination / "PHASE_CLOSEOUT_hosted.json", {
                "plan_sha256": "f" * 64, "consumer_finished": True,
                "root_window_release_confirmed": False, "request_window_released": False})
            runner.claim_security500_grant(destination, "f" * 64, "local", consumer)
            with self.assertRaises(FileExistsError):
                runner.claim_security500_grant(destination, "f" * 64, "local", consumer)

    def test_rejected_window_latches_stop_before_lock(self):
        consumer = self.consumer()
        consumer.validate_request.side_effect = ValueError("revoked")
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            with patch.object(runner, "LOCK", destination / "claim.lock"):
                with self.assertRaisesRegex(ValueError, "revoked"):
                    with runner.security500_request_window(destination, "f" * 64, consumer, route="base"):
                        self.fail("revoked grant entered")
            consumer.stop.assert_called_once()
            self.assertTrue((destination / "SECURITY500_COLLECTION_STOP.json").exists())
            self.assertFalse((destination / "claim.lock").exists())

    def test_child_context_is_private_and_mixed_consumers_rejected(self):
        consumer = self.consumer()
        process = Mock()
        process.wait.return_value = process.poll.return_value = 0
        batch = {"rules_version": "synthetic", "seed": 1, "models": ["qwen_base"]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner.subprocess, "Popen", return_value=process) as start, \
                patch.object(runner, "audit_batch", return_value=([], {})):
            runner.run_batch(batch, Path(tmp) / "batch_01", Path("unused"), security500_consumer=consumer)
            self.assertEqual(start.call_args.kwargs["env"][runner.SECURITY500_CONTEXT_ENV], "unit-private-scope")
            self.assertNotIn("unit-private-scope", start.call_args.args[0])
            consumer.private_child_context.assert_called_once_with("f" * 64, source_scope={
                "plan_path": str((Path(tmp) / "PLAN.json").resolve()), "batch_index": 0})
            runner.run_batch(batch, Path(tmp) / "batch_02", None, security500_consumer=consumer)
            self.assertNotIn("--credentials", start.call_args.args[0])
            with self.assertRaisesRegex(ValueError, "batch identity"):
                runner.run_batch(batch, Path(tmp) / "batch_1", Path("unused"), security500_consumer=consumer)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                runner.run_batch(batch, Path(tmp) / "batch2", Path("unused"), handoff_consumer=Mock(), security500_consumer=consumer)

    def test_new_model_cannot_fall_back_to_legacy_window(self):
        candidate = {"models": ["sp_lense_security500"], "episodes": 0, "batches": []}
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "plan", return_value=candidate), \
                patch.object(runner, "provider352_planning_core_ref", return_value=None), \
                patch.object(runner, "security500_planning_core_ref", return_value=None), \
                patch.object(runner, "request_window", side_effect=AssertionError("legacy fallback")), \
                patch.object(runner, "write_json") as write:
            with self.assertRaisesRegex(ValueError, "planning core and request scope"):
                runner.execute_locked(Path(tmp), Path("unused"), Path("unused"), "all")
            write.assert_not_called()

    def test_planning_core_without_consumer_denies_before_plan_or_output(self):
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "security500_planning_core_ref", return_value=consumer.core_binding), \
                patch.object(runner, "security500_active_consumer", side_effect=ValueError("NOT_READY")), \
                patch.object(runner, "plan") as plan, patch.object(runner, "write_json") as write:
            with self.assertRaisesRegex(ValueError, "NOT_READY"):
                runner.execute_locked(Path(tmp), Path("unused"), Path("unused"), "all")
            plan.assert_not_called()
            write.assert_not_called()

    def test_changed_source_stops_before_batch(self):
        candidate = {"models": ["qwen_base_neutral", "qwen_base", "sp_lense_security500_neutral",
                                "sp_lense_security500", "terra", "deepseek"], "episodes": 1,
                     "batches": [{"models": ["qwen_base"], "seed": 1}],
                     "seeds": [], "protocol": "unit", "horizon_hours": 48}
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            runner.write_json(destination / "PLAN.json", candidate)
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "provider352_planning_core_ref", return_value=None), \
                    patch.object(runner, "security500_planning_core_ref", return_value=consumer.core_binding), \
                    patch.object(runner, "plan_inputs_match", return_value=False), \
                    patch.object(runner, "LOCK", destination / "claim.lock"), \
                    patch.object(runner, "request_window", side_effect=AssertionError("legacy fallback")), \
                    patch.object(runner, "run_batch") as dispatch:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runner.execute_locked(destination, Path("unused"), Path("unused"), "all", security500_consumer=consumer)
            dispatch.assert_not_called()
            consumer.stop.assert_called()
            self.assertTrue((destination / "SECURITY500_COLLECTION_STOP.json").exists())

    def test_abnormal_hosted_child_or_summary_write_latches_scope(self):
        candidate = {"models": ["qwen_base_neutral", "qwen_base", "sp_lense_security500_neutral",
                                "sp_lense_security500", "terra", "deepseek"], "episodes": 1,
                     "batches": [{"models": ["terra", "deepseek"], "seed": 1}],
                     "seeds": [], "protocol": "unit", "horizon_hours": 48}
        for index, failure in enumerate((KeyboardInterrupt(), OSError("summary write failed"))):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)
                consumer = self.consumer()
                consumer.pins["expected_window_sha256"] = ("c" if index == 0 else "d") * 64
                runner.write_json(destination / "PLAN.json", candidate)
                completed = ([{"evidence_status": "audited_complete"}], {"batch": "batch_01", "exit_code": 0})
                with patch.object(runner, "plan", return_value=candidate), \
                        patch.object(runner, "provider352_planning_core_ref", return_value=None), \
                        patch.object(runner, "security500_planning_core_ref", return_value=consumer.core_binding), \
                        patch.object(runner, "plan_inputs_match", return_value=True), \
                        patch.object(runner, "source_matches", return_value=True), \
                        patch.object(runner, "run_batch", side_effect=failure if isinstance(failure, KeyboardInterrupt) else None,
                                     return_value=completed), \
                        patch.object(runner, "summarize", side_effect=failure if isinstance(failure, OSError) else None):
                    with self.assertRaises(type(failure)):
                        runner.execute_locked(destination, Path("unused"), Path("unused"), "all", security500_consumer=consumer)
                consumer.stop.assert_called()
                self.assertTrue((destination / "SECURITY500_COLLECTION_STOP.json").exists())

    def test_cli_requires_all_independent_security500_pins(self):
        args = ["comparison", "--output", "unused", "--security500-core-path", ".local/core.json",
                "--security500-core-sha256", "a" * 64, "--security500-review-sha256", "b" * 64]
        with patch.object(runner.sys, "argv", args), patch.object(runner, "plan", return_value={}) as plan, \
                patch.object(runner, "print", create=True):
            runner.main()
        self.assertEqual(plan.call_args.kwargs["security500_core_binding"], {
            "core_binding": {"path": ".local/core.json", "sha256": "a" * 64}, "trusted_review_sha256": "b" * 64})
        with patch.object(runner.sys, "argv", args[:-2]), self.assertRaises(SystemExit):
            runner.main()


if __name__ == "__main__":
    unittest.main()
