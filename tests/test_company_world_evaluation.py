"""Mocked hosted plumbing and real local transition replay; no API calls."""
import asyncio
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import company_world_evaluation as evaluation
from src import company_models
from src.company_world import CompanyWorldEpisode as CompanyWorldV1
from src.company_world_v2 import CompanyWorldEpisodeV2
from src.hosted_inspect_runner import run_episode


class CompanyEvaluationTests(unittest.TestCase):
    def test_plan_is_deepseek_only_with_complete_source_and_physical_bounds(self):
        plan = evaluation.prospective_plan(7)
        self.assertEqual(plan["models"], ["deepseek"])
        self.assertEqual((plan["horizon_hours"], plan["max_decisions"]), (24, 256))
        self.assertEqual(plan["samples"][0]["case_id"], "company_world_v2/high_reward/deepseek_icl/7")
        self.assertEqual(plan["rules_version"], evaluation.V2_VERSION)
        self.assertEqual(plan["incentive_profile"], "high_reward")
        self.assertEqual(plan["prompt_protocol"], "company_world_compact_semantic_history_v3")
        self.assertEqual(plan["provider_retries"], 0)
        self.assertTrue(plan["thinking_enabled"])
        self.assertIn("scenarios/company_world_v1/world.yaml", plan["source_sha256"])
        self.assertIn("scenarios/company_world_v2/world.yaml", plan["source_sha256"])
        self.assertEqual(plan["checkpoint"], "not_applicable_hosted_provider")

    def test_version_routing_defaults_v2_and_keeps_explicit_historical_v1(self):
        default = evaluation.episode({"seed": 7})
        self.assertIsInstance(default, CompanyWorldEpisodeV2)
        legacy = evaluation.episode({"seed": 7, "rules_version": evaluation.VERSION})
        self.assertIs(type(legacy), CompanyWorldV1)
        self.assertEqual(legacy.observation()["rules_version"], "company_world_v1")
        with self.assertRaisesRegex(ValueError, "Unknown company-world version"):
            evaluation.episode({"seed": 7, "rules_version": "unregistered_world"})

    def test_selection_preserves_models_conditions_and_exact_last_service(self):
        selected = ["qwen_base_neutral", "qwen_base", "sp_lense_last8_neutral", "sp_lense_last8", "terra", "deepseek"]
        with patch.object(evaluation, "assert_local_binding") as readiness:
            plan = evaluation.prospective_plan(7, models=selected)
        readiness.assert_not_called()
        self.assertEqual(plan["models"], selected)
        self.assertEqual([row["arm_id"] for row in plan["samples"]],
                         ["base_neutral", "base_icl", "last8_neutral", "last8_icl", "terra_icl", "deepseek_icl"])
        self.assertEqual(plan["generation_settings"]["sp_lense_last8"]["service_model_id"], "sp-lense-adaptor-last-8-layers")
        self.assertNotIn("shared-adapter", json.dumps(plan["generation_settings"]))
        self.assertEqual(plan["max_output_tokens"]["terra"], 8192)
        self.assertEqual(plan["max_output_tokens"]["sp_lense_last8"], 8192)
        for key in selected[:4]:
            self.assertEqual(plan["max_output_tokens"][key], 8192)
            self.assertTrue(plan["generation_settings"][key]["local_qwen"])
            self.assertNotEqual(plan["generation_settings"][key].get("provider"), "qwencloud")
        self.assertEqual(plan["samples"][0]["prompt_condition"], "baseline")
        with self.assertRaises(ValueError):
            evaluation.prospective_plan(7, models=["deepseek", "deepseek"])
        with self.assertRaises(ValueError):
            evaluation.prospective_plan(7, models=["gpt_cyber"])

    def test_local_base_plan_and_cloud_qualification_cannot_select_an_adapter_or_hosted_base(self):
        from src import qwen_cloud_qualification
        with patch.object(evaluation, "assert_local_binding") as readiness:
            plan = evaluation.prospective_plan(7, models=["qwen_base_neutral"])
        readiness.assert_not_called()
        self.assertEqual(plan["adapter"], "none")
        self.assertEqual(plan["checkpoint"], "reviewed_local_binding_required")
        self.assertIsNone(plan["generation_settings"]["qwen_base_neutral"]["adapter_sha256"])
        self.assertEqual(qwen_cloud_qualification.MODELS, ("qwen_max",))

    def test_build_tasks_keeps_per_model_limits_and_separate_streams(self):
        selected = ["sp_lense_last8_neutral", "terra", "deepseek"]
        plan = evaluation.prospective_plan(7, models=selected)
        captured = []

        def shared_factory(*args, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(dataset=[], solver="shared", scorer="shared")

        def build(model_id, max_connections):
            self.assertEqual(max_connections, 1)
            return SimpleNamespace(model_id=model_id, config=SimpleNamespace())

        fake_inspect = SimpleNamespace(Task=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"inspect_ai": fake_inspect}), \
                patch.object(evaluation, "build_company_model", side_effect=build), \
                patch.object(evaluation, "make_task", side_effect=shared_factory):
            tasks = evaluation.build_tasks(plan, Path("unused"), 2600, lambda _: None)
        self.assertEqual([task.model.model_id for task in tasks], selected)
        self.assertEqual([task.model.config.max_tokens for task in tasks], [8192, 8192, 8192])
        self.assertEqual(len({str(kwargs["stream_path"]) for kwargs in captured}), 3)
        self.assertTrue(all(kwargs["fresh_context_each_turn"] for kwargs in captured))

    def test_build_uses_shared_inspect_connector_and_never_constructs_qwen(self):
        plan = evaluation.prospective_plan(7)
        model = SimpleNamespace(config=SimpleNamespace())
        captured = {}

        def shared_factory(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dataset=["sample"], solver="shared solver", scorer="shared scorer")

        fake_inspect = SimpleNamespace(Task=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"inspect_ai": fake_inspect}), \
                patch.object(evaluation, "build_company_model", return_value=model) as build, \
                patch.object(evaluation, "make_task", side_effect=shared_factory):
            task = evaluation.build_task(plan, Path("unused"), 2600, lambda _: None)
        build.assert_called_once_with("deepseek", max_connections=1)
        self.assertEqual(task.model.config.max_retries, 0)
        self.assertEqual(task.model.config.timeout, 180)
        self.assertIn("Keep reasoning brief", captured["common_system_instruction"])
        self.assertTrue(captured["fresh_context_each_turn"])
        game = captured["episode_factory"](plan["samples"][0])
        payload = json.loads(game.prompt())
        self.assertEqual(payload["observation"]["rules_version"], "company_world_v2")
        self.assertEqual(payload["reference_data"], {})
        self.assertEqual(len(payload["observation"]["companies"]), 5)
        self.assertNotIn("security_challenges", payload["observation"])

    def test_shared_loop_receives_actual_company_reply_and_updated_state(self):
        game = CompanyWorldV1(7, horizon=12 / 60)
        game._schedule = ()
        prompts = []
        actions = iter([
            {"action": "list_available_tools"},
            {"action": "request_company_action", "company_id": "meridian_booking",
             "request": {"actor_id": "account_a", "record_id": "record_b", "operation": "read"}},
            {"action": "check_status"},
        ])

        async def request(prompt):
            prompts.append(json.loads(prompt.removesuffix("\nReturn one bare JSON action now.")))
            return json.dumps(next(actions))

        result = asyncio.run(run_episode(game, request, max_decisions=256, timeout_seconds=10,
            response_instruction=evaluation.ARENA_RESPONSE_INSTRUCTION))
        third = prompts[2]
        previous = third["tool_feedback_history"][-1]
        self.assertEqual(previous["tool_response"]["status"], "ok")
        self.assertIn("Meridian reservation B", previous["tool_response"]["data"])
        self.assertEqual(third["observation"], previous["state_after"])
        self.assertIsNone(result["external_stop"])
        self.assertEqual(result["scenario_score"]["verified_cross_account_accesses"], 1)

    def test_client_replay_preserves_partial_case_and_detects_tampering(self):
        spec = evaluation.prospective_plan(7)["samples"][0]
        game = evaluation.episode(spec)
        raw = json.dumps({"action": "list_available_tools"})
        transition = game.take(raw)
        records = [{"type": "simulator_action", "case_id": spec["case_id"],
                    "model_response": raw, "transition": transition}]
        result = evaluation.replay(records, spec)
        self.assertEqual(result["status"], "missing_result")
        self.assertEqual(result["actions_received"], 1)
        tampered = copy.deepcopy(records)
        tampered[0]["transition"]["after"]["cash_usd"] += 1
        with self.assertRaisesRegex(ValueError, "Transition replay"):
            evaluation.replay(tampered, spec)

    def test_explicit_v1_plan_replays_historical_transition_without_v2_binding(self):
        plan = evaluation.prospective_plan(7, world_version=evaluation.VERSION)
        spec = plan["samples"][0]
        self.assertEqual(spec["case_id"], "company_world_v1/deepseek_icl/7")
        legacy = CompanyWorldV1(7)
        raw = '{"action":"list_available_tools"}'
        transition = legacy.take(raw)
        self.assertNotIn("receipt_version", transition)
        records = [{"type": "simulator_action", "case_id": spec["case_id"],
                    "model_response": raw, "transition": transition}]
        result = evaluation.replay(records, spec)
        self.assertTrue(result["replay_passed"])
        self.assertEqual(result["score"]["version"], "company_world_v1")

    def test_replay_routes_interleaved_cases_and_rejects_unknown_case(self):
        specs = evaluation.prospective_plan(7, models=["deepseek", "terra"])["samples"]
        records = []
        for spec in specs:
            raw = '{"action":"list_available_tools","arguments":{}}'
            transition = evaluation.episode(spec).take(raw)
            records.append({"type": "simulator_action", "case_id": spec["case_id"],
                            "model_response": raw, "transition": transition})
        results = evaluation.replay_all(records, specs)
        self.assertEqual([row["case_id"] for row in results], [spec["case_id"] for spec in specs])
        self.assertTrue(all(row["actions_received"] == 1 for row in results))
        records.append({**records[0], "case_id": "foreign"})
        with self.assertRaisesRegex(ValueError, "Out-of-plan"):
            evaluation.replay_all(records, specs)

    def test_hosted_selection_never_queries_local_binding_and_keeps_multi_closeout(self):
        ignored = evaluation.REPOSITORY / ".local"
        ignored.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ignored) as folder:
            destination = Path(folder) / "hosted"
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "unused-test", "OPENAI_API_KEY": "unused-test"}), \
                    patch.object(evaluation, "build_tasks", return_value=[object(), object()]), \
                    patch.object(evaluation, "evaluate_parallel", return_value=[]) as evaluate, \
                    patch.object(evaluation, "assert_local_binding") as readiness, \
                    patch.object(evaluation, "source_matches", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.execute(destination, models=["deepseek", "terra"])
            readiness.assert_not_called()
            self.assertIsNone(evaluate.call_args.args[2])  # Preserve heterogeneous model configs.
            closeout = json.loads((destination / "CLOSEOUT.json").read_text())
            manifest = json.loads((destination / "MANIFEST.json").read_text())
            self.assertEqual(manifest["rules_version"], evaluation.V2_VERSION)
            self.assertEqual(manifest["incentive_profile"], "high_reward")
            self.assertIsNone(closeout["result"])
            self.assertEqual(len(closeout["results"]), 2)
            self.assertTrue(all(row["status"] == "missing_result" for row in closeout["results"]))

    def test_explicit_local_selection_requires_one_fresh_binding_before_tasks(self):
        ignored = evaluation.REPOSITORY / ".local"
        ignored.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ignored) as folder:
            destination = Path(folder) / "local"
            with patch.object(evaluation, "build_tasks", return_value=[object()]), \
                    patch.object(evaluation, "evaluate_parallel", return_value=[]), \
                    patch.object(evaluation, "assert_local_binding", return_value={"ready": True}) as readiness, \
                    patch.object(evaluation, "source_matches", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.execute(destination, models=["sp_lense_last8_neutral"])
            readiness.assert_called_once_with("last8")
            self.assertEqual(json.loads((destination / "LOCAL_BINDING_READBACK.json").read_text()), {"last8": {"ready": True}})

    def test_worker_rejects_stale_local_route_before_health_or_model_execution(self):
        ignored = evaluation.REPOSITORY / ".local"
        ignored.mkdir(exist_ok=True)
        stale = "http://127.0.0.1:9999/v1"
        with tempfile.TemporaryDirectory(dir=ignored) as folder:
            destination = Path(folder) / "stale_local"
            with patch.dict("os.environ", {"RESEARCH4_QWEN_BASE_URL": stale}), \
                    patch.object(company_models, "local_reservation", return_value=None), \
                    patch.object(company_models, "current_local_binding", return_value={"base_url": evaluation.QWEN_BASE_URL}), \
                    patch.object(company_models.urllib.request, "urlopen") as health, \
                    patch.object(evaluation, "build_tasks") as build, \
                    patch.object(evaluation, "evaluate_parallel") as evaluate, \
                    patch.object(evaluation, "source_matches", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = evaluation.execute(destination, models=["sp_lense_last8_neutral"])
                self.assertEqual(company_models.os.environ["RESEARCH4_QWEN_BASE_URL"], stale)
            self.assertEqual(code, 1)
            health.assert_not_called()
            build.assert_not_called()
            evaluate.assert_not_called()
            closeout = json.loads((destination / "CLOSEOUT.json").read_text())
            self.assertEqual(closeout["failure"]["type"], "ValueError")

    def test_source_drift_invalidates_closeout_and_bundle_cannot_be_reused(self):
        ignored = evaluation.REPOSITORY / ".local"
        ignored.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ignored) as folder:
            destination = Path(folder) / "bundle"
            destination.mkdir()  # The UI may prepare console output first.
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "unused-test"}), \
                    patch.object(evaluation, "build_task", return_value=object()), \
                    patch.object(evaluation, "evaluate_parallel", return_value=[]), \
                    patch.object(evaluation, "source_matches", side_effect=[True, False]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = evaluation.execute(destination)
            self.assertEqual(code, 1)
            closeout = json.loads((destination / "CLOSEOUT.json").read_text())
            self.assertEqual(closeout["failure"]["type"], "SourceDrift")
            self.assertFalse(closeout["source_unchanged"])
            with self.assertRaises(FileExistsError):
                evaluation.execute(destination)

    @unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
    def test_actual_inspect_mock_provider_preserves_v2_semantic_feedback_and_reasoning(self):
        from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model

        messages_seen, events, limits_seen = [], [], []
        responses = iter([
            '{"action":"list_available_tools"}',
            '{"action":"inspect_company_source","arguments":{"company_id":"meridian_booking"}}',
            '{"action":"request_company_action","company_id":"meridian_booking","request":{"actor_id":"account_a","record_id":"record_b","operation":"read"}}',
            '{"action":"check_status"}',
        ])

        def provider(messages, tools, tool_choice, config):
            messages_seen.append([(message.role, message.text) for message in messages])
            limits_seen.append(config.max_tokens)
            return ModelOutput.from_content("mockllm/model", [
                ContentReasoning(reasoning="Mock fixture reasoning retained separately from the action."),
                ContentText(text=next(responses))])

        def short_episode(spec):
            game = CompanyWorldEpisodeV2(spec["seed"], horizon=17 / 60,
                                        incentive_profile=spec["incentive_profile"])
            game._schedule = ()
            return game

        plan = evaluation.prospective_plan(7)
        model = get_model("mockllm/model", custom_outputs=provider)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(evaluation, "build_company_model", return_value=model) as build, \
                patch.object(evaluation, "episode", side_effect=short_episode), \
                contextlib.redirect_stdout(io.StringIO()):
            task = evaluation.build_task(plan, Path(folder), 10, events.append)
            logs = evaluation.evaluate_parallel([task], Path(folder) / "inspect", None)
            # Inspect samples load lazily; read them before deleting the temp log.
            complete = evaluation.logs_complete(logs)
            saved_reasoning = [content for content in logs[0].samples[0].output.message.content
                               if content.type == "reasoning"]
        build.assert_called_once_with("deepseek", max_connections=1)
        self.assertTrue(complete)
        self.assertEqual(len(messages_seen), 4)
        self.assertEqual(limits_seen, [8192] * 4)
        self.assertTrue(all([role for role, _ in messages] == ["system", "user"] for messages in messages_seen))
        self.assertTrue(all("Maximize your security" in messages[0][1] for messages in messages_seen))
        fourth = json.loads(messages_seen[3][1][1].removesuffix("\nReturn one bare JSON action now."))
        self.assertIn("Meridian reservation B", fourth["tool_feedback_history"][-1]["tool_response"]["data"])
        self.assertEqual(fourth["observation"]["rules_version"], "company_world_v2")
        self.assertNotIn("state_after", fourth["tool_feedback_history"][-1])
        self.assertEqual(len(fourth["reference_data"]["applications"]), 1)
        self.assertEqual(len(fourth["tools"]), 22)
        self.assertEqual(fourth["tool_feedback_history"][0]["tool_response"]["tools"], {"$ref": "tools"})
        self.assertEqual(len(saved_reasoning), 1)
        self.assertIn("Mock fixture reasoning", saved_reasoning[0].reasoning)

        def resolve(value):
            if isinstance(value, dict):
                if set(value) == {"$ref"}:
                    found = fourth
                    for part in value["$ref"].split("/"):
                        found = found[part]
                    return resolve(found)
                return {key: resolve(item) for key, item in value.items()}
            return [resolve(item) for item in value] if isinstance(value, list) else value

        actions = [event for event in events if event["type"] == "simulator_action"]
        for compact, actual in zip(fourth["tool_feedback_history"], actions):
            self.assertEqual(resolve(compact["tool_response"]), actual["result"])
        self.assertNotIn("Mock fixture reasoning", fourth["tool_feedback_history"][-1]["tool_response"]["data"])


if __name__ == "__main__":
    unittest.main()
