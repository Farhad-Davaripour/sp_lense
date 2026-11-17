"""Fresh-candidate matrix and dispatch wiring; synthetic CPU fixtures only."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import infrastructure_preservation_comparison as runner


class IndependentStateComparisonTests(unittest.TestCase):
    def config(self):
        ids = list(runner.INDEPENDENT_STATE_IDS)
        return {"schema": runner.INDEPENDENT_STATE_GAME_SCHEMA, "role": "unit_test_not_actual_cohort",
            "seeds": [101, 102, 103], "local_models": ids, "hosted_models": [],
            "world_version": "company_world_long_horizon_v1", "incentive_profile": "high_reward",
            "horizon_hours": 48, "outer_seconds": 14400, "generation_seed": 20261108,
            "generation_profile_sha256": runner.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
            "history_mode": runner.FULL_HISTORY_MODE, "prompt_protocol": runner.FULL_HISTORY_PROMPT_PROTOCOL,
            "user_task_instruction": runner.FULL_HISTORY_USER_TASK,
            "execution_order": [{"world_index": i, "model_id": name}
                                for i, name in runner.INDEPENDENT_STATE_EXECUTION_ORDER],
            "primary_outcome": "survival48h", "primary_contrast": [ids[2], ids[0]],
            "same_prompt_adapter_contrasts": [[ids[2], ids[0]], [ids[3], ids[1]]],
            "comparison_contrasts": [[ids[2], ids[0]], [ids[3], ids[1]], [ids[1], ids[0]], [ids[3], ids[2]]]}

    def prospective(self, calls):
        def stub(seed, *, models, **kwargs):
            calls.append((seed, list(models), kwargs))
            return {"seed": seed, "models": models,
                "samples": [{"case_id": f"unit/{models[0]}/{seed}", "model_id": models[0], "seed": seed}],
                "max_decisions": 256, "horizon_hours": 48, "outer_seconds": 14400,
                "history_mode": kwargs["history_mode"], "prompt_protocol": runner.FULL_HISTORY_PROMPT_PROTOCOL}
        return stub

    def consumer(self, grants):
        value = object.__new__(runner.IndependentStateGameConsumer)
        value.contract = SimpleNamespace(BASE="qwen3.8-27b-base", FT="sp-20261116-01-aaaaaaaa",
            ARMS=("qwen3.8-27b-base", "sp-20261116-01-aaaaaaaa"))
        value.core_binding = {"core_binding": {"path": ".local/synthetic/core.json", "sha256": "a" * 64},
                              "trusted_review_sha256": "b" * 64}
        value.source_binding = {"synthetic_fixture": True}
        value.pins = {"expected_game_plan_sha256": "c" * 64, "expected_window_sha256": "d" * 64}
        value._grant_directory = Mock(return_value=grants)
        value.remaining_global_seconds = Mock(return_value=100)
        value.private_child_context = Mock(return_value="synthetic-five-field-context")
        value.stop = Mock()
        return value

    def test_only_exact_four_new_roles_without_hosted_or_historical_aliases(self):
        ids = runner.INDEPENDENT_STATE_IDS
        self.assertEqual(runner.validate_model_matrix(ids, ()), (ids, ()))
        for names, hosted in ((ids, ("terra",)), (ids[::-1], ()),
                              ((*ids[:2], "sp_lense_transfer_neutral", ids[3]), ()),
                              ((*ids, ids[0]), ())):
            with self.subTest(names=names, hosted=hosted), self.assertRaises(ValueError):
                runner.validate_model_matrix(names, hosted)

    def test_candidate_only_plan_has_twelve_fixed_cases_and_fresh3072_budget(self):
        calls = []
        binding = {"synthetic_candidate_fixture": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self.config()), encoding="utf-8")
            with patch.object(runner, "register_independent_state_candidate") as register, \
                    patch.object(runner, "initialize_independent_state_source", side_effect=AssertionError("late source")), \
                    patch.object(runner, "initialize_independent_state_core", side_effect=AssertionError("late core")), \
                    patch.object(runner, "selected_models"), \
                    patch.object(runner, "prospective_plan", side_effect=self.prospective(calls)):
                frozen = runner.plan(path, independent_state_candidate_binding=binding)
        register.assert_called_once_with(binding)
        self.assertEqual(frozen["episodes"], 12)
        self.assertEqual([(seed, names[0]) for seed, names, _ in calls],
                         [(101 + i, name) for i, name in runner.INDEPENDENT_STATE_EXECUTION_ORDER])
        self.assertEqual(len({(seed, names[0]) for seed, names, _ in calls}), 12)
        self.assertTrue(all(kw["history_mode"] == runner.FULL_HISTORY_MODE and
                            kw["user_task_instruction"] == runner.FULL_HISTORY_USER_TASK for _, _, kw in calls))
        self.assertEqual((frozen["transport_max_total_native_dispatches"],
                          frozen["transport_max_additional_native_dispatches"]), (3072, 3072))
        self.assertNotIn("independent_state_source_binding", frozen)
        self.assertIn("fresh direct", frozen["transport_budget_note"])

    def test_bad_history_task_order_seed_count_or_old_schema_rejected_before_world_planning(self):
        changes = [("history_mode", None), ("user_task_instruction", "different"),
                   ("prompt_protocol", "old"), ("schema", runner.CONTINUITY_TRANSFER_FULL_HISTORY_SCHEMA),
                   ("seeds", list(range(9))), ("generation_seed", True)]
        for field, value in changes:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                config = self.config(); config[field] = value
                path = Path(tmp) / "bad.json"; path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(runner, "prospective_plan", side_effect=AssertionError("world planning")), \
                        self.assertRaises(ValueError):
                    runner.plan(path)
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(); config["execution_order"][0]["world_index"] = False
            path = Path(tmp) / "bad.json"; path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "prospective_plan", side_effect=AssertionError("world planning")), self.assertRaises(ValueError):
                runner.plan(path)

    def test_candidate_cannot_mix_historical_core_before_any_registration(self):
        with patch.object(runner, "register_independent_state_candidate") as register, self.assertRaises(ValueError):
            runner.plan("unused", independent_state_candidate_binding={}, continuity_transfer_core_binding={})
        register.assert_not_called()

    def test_unqualified_execute_refuses_before_destination_or_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"; path.write_text(json.dumps(self.config()), encoding="utf-8")
            with patch.dict(runner.os.environ, {}, clear=True), \
                    patch.object(runner, "_execute_in_destination", side_effect=AssertionError("execution")), \
                    self.assertRaisesRegex(ValueError, "actual qualified core"):
                runner.execute(Path(tmp) / "not_created", None, path)
            self.assertFalse((Path(tmp) / "not_created").exists())

    def test_execute_uses_new_core_initializer_and_preserves_source_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            consumer = self.consumer(Path(tmp) / "grants")
            with patch.dict(runner.os.environ, {}, clear=True), \
                    patch.object(runner, "initialize_independent_state_core") as initialize, \
                    patch.object(runner, "initialize_continuity_transfer_core", side_effect=AssertionError("historical core")), \
                    patch.object(runner, "initialize_security500_core", side_effect=AssertionError("historical core")), \
                    patch.object(runner, "_execute_in_destination") as execute:
                runner.execute(Path(tmp) / "out", None, "unused", independent_state_consumer=consumer)
            initialize.assert_called_once_with(consumer.core_binding, source_binding=consumer.source_binding)
            self.assertIs(execute.call_args.kwargs["security500_consumer"], consumer)
            self.assertEqual(runner._two_route_prefix(consumer), "INDEPENDENT_STATE")
            self.assertEqual(runner._two_route_grants(consumer), Path(tmp) / "grants")

    def test_mismatched_qualified_core_stops_before_claim_or_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            consumer = self.consumer(Path(tmp) / "grants")
            candidate = {"models": runner.INDEPENDENT_STATE_IDS}
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "continuity_transfer_planning_core_ref", return_value=None), \
                    patch.object(runner, "independent_state_planning_core_ref", return_value={"wrong": "core"}), \
                    patch.object(runner, "claim_security500_grant", side_effect=AssertionError("claim")), \
                    patch.object(runner, "run_batch", side_effect=AssertionError("child")), \
                    self.assertRaisesRegex(ValueError, "exact new qualified planning core"):
                runner._execute_locked(Path(tmp), None, "unused", "all", security500_consumer=consumer)
            self.assertFalse((Path(tmp) / "PLAN.json").exists())

    def test_direct_new_alias_audit_still_applies_late_child_failure_gate(self):
        # No world is reconstructed: a synthetic empty replay isolates dispatch
        # to the established operational-error classifier for either new route.
        batch = {"models": [runner.INDEPENDENT_STATE_IDS[0]], "samples": []}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "batch_01"; directory.mkdir()
            runner.write_json(directory / "MANIFEST.json", {})
            runner.write_json(directory / "CLOSEOUT.json", {"source_unchanged": True,
                "results": [], "failure": {"type": "synthetic_late_error"}})
            with patch.object(runner, "validate_batch_manifest"), patch.object(runner, "replay_all", return_value=[]), \
                    patch.object(runner, "_continuity_operationally_incomplete", return_value=[]) as classify:
                runner.audit_batch(batch, directory, 1)
            self.assertEqual(classify.call_args.args[1], {"batch": "batch_01", "exit_code": 1,
                "failure": {"type": "synthetic_late_error"}})

    def test_child_uses_new_context_and_global_timeout_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            consumer = self.consumer(Path(tmp) / "grants")
            process = Mock(); process.wait.return_value = 0; process.poll.return_value = 0
            batch = {"seed": 101, "rules_version": "company_world_long_horizon_v1",
                     "models": [runner.INDEPENDENT_STATE_IDS[0]], "horizon_hours": 48,
                     "outer_seconds": 14400, "history_mode": runner.FULL_HISTORY_MODE,
                     "user_task_instruction": runner.FULL_HISTORY_USER_TASK}
            with patch.dict(runner.os.environ, {}, clear=True), \
                    patch.object(runner.subprocess, "Popen", return_value=process) as spawn, \
                    patch.object(runner, "audit_batch", return_value=([], {})) as audit:
                runner.run_batch(batch, Path(tmp) / "batch_01", None, security500_consumer=consumer)
            environment = spawn.call_args.kwargs["env"]
            self.assertEqual(environment[runner.INDEPENDENT_STATE_CONTEXT_ENV], "synthetic-five-field-context")
            self.assertNotIn(runner.CONTINUITY_TRANSFER_CONTEXT_ENV, environment)
            self.assertNotIn(runner.SCIENTIFIC_AMENDMENT_ENV, environment)
            consumer.private_child_context.assert_called_once_with("c" * 64,
                source_scope={"plan_path": str((Path(tmp) / "PLAN.json").resolve()), "batch_index": 0})
            process.wait.assert_called_once_with(timeout=105)
            audit.assert_called_once_with(batch, Path(tmp) / "batch_01", 0, continuity_transfer=True)

    def test_ledger_binds_new_candidate_and_retains_all_twelve_unknown_slots(self):
        config = self.config(); calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"; path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "selected_models"), patch.object(runner, "prospective_plan", side_effect=self.prospective(calls)):
                frozen = runner.plan(path)
            runner.write_json(Path(tmp) / "PLAN.json", frozen)
            new_ft = {"adapter_sha256": "1" * 64, "adapter_config_sha256": "2" * 64}
            with patch.dict(runner.MODEL_SPECS, {runner.INDEPENDENT_STATE_IDS[2]: new_ft}):
                runner.write_continuity_transfer_ledger(Path(tmp), frozen, [])
            ledger = runner.read_json(Path(tmp) / "CANDIDATE_COHORT_LEDGER.json")
        self.assertEqual(ledger["candidate_adapter_sha256"], "1" * 64)
        self.assertEqual(ledger["candidate_config_sha256"], "2" * 64)
        self.assertEqual(len(ledger["slots"]), 12)
        self.assertTrue(all(row["status"] == "NOT_STARTED_UNKNOWN" for row in ledger["slots"]))
        self.assertFalse(ledger["all_twelve_audited_known"])
        self.assertIn("independent-state", ledger["schema"])


if __name__ == "__main__":
    unittest.main()
