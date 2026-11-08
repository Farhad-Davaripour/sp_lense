"""Local tests of the hosted adapter. They do not call an API."""
import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "hosted_inspect_runner.py"
spec = importlib.util.spec_from_file_location("hosted_inspect_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeSimulator:
    def __init__(self):
        self.decisions = 0
        self.termination_reason = None
        self.received = []
        self.feedback = []

    def prompt(self):
        return json.dumps({"task": "complete the task", "tools": ["list_available_tools"],
                           "observation": {"decisions": self.decisions},
                           "tool_feedback_history": self.feedback})

    def take(self, raw):
        self.received.append(raw)
        self.decisions += 1
        if raw == '{"action":"complete"}':
            self.termination_reason = "verified_task_goal"
        action = json.loads(raw).get("action") if raw.startswith('{') else None
        result = ({"tools": ["read_order_records", "complete"]} if action == "list_available_tools"
                  else {"records": [{"quantity": 3, "price": 12}]} if action == "read_order_records"
                  else "recorded")
        self.feedback.append({"action": action, "tool_response": result})
        return {"action_id": raw, "result": "recorded"}

    def finish(self, reason):
        self.termination_reason = reason

    def score(self):
        return {"task_complete": self.termination_reason == "verified_task_goal",
                "actual_model_decisions": self.decisions,
                "termination_reason": self.termination_reason}


class HostedLoopTests(unittest.IsolatedAsyncioTestCase):
    def test_local_qwen_models_use_named_openai_compatible_provider(self):
        for mode in ("base", "shared-adapter"):
            model = f"openai-api/research4-qwen/qwen3.8-27b-{mode}"
            config = runner.hosted_model_config(model)
            self.assertEqual(config["model"], model)
            self.assertEqual(config["model_args"], {"responses_api": False, "stream": True})
            self.assertNotIn("reasoning_effort", config)
            self.assertEqual(config["extra_body"],
                             {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(runner.reasoning_kind_for_model(model), "raw")

    async def test_deepseek_reasoning_is_not_mislabeled_as_openai_summary(self):
        events = []
        recorder = runner.StreamRecorder("seed:7", 0, on_event=events.append,
                                          reasoning_kind="raw")
        await recorder(SimpleNamespace(type="reasoning", reasoning="Selecting a tool."))
        self.assertEqual(events[0]["event"], "reasoning_delta")
        self.assertEqual(events[0]["reasoning_kind"], "raw")

    def test_telemetry_failure_cannot_change_a_model_result(self):
        def broken_display(record):
            raise RuntimeError("display unavailable")

        with self.assertWarns(RuntimeWarning):
            runner.notify_event(broken_display, {"type": "episode_result"})
        original = {"nested": {"task_complete": True}}
        runner.notify_event(lambda item: item["nested"].update(task_complete=False), original)
        self.assertTrue(original["nested"]["task_complete"])

    def test_module_cli_help_starts_without_credentials_or_model_calls(self):
        result = subprocess.run([sys.executable, "-m", "src.hosted_inspect_runner", "--help"],
                                cwd=MODULE_PATH.parents[1], capture_output=True, text=True,
                                timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--fixture-dir", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--prompt-condition", result.stdout)

    async def test_stream_deltas_are_saved_but_only_final_response_is_dispatched(self):
        sim = FakeSimulator()
        state = SimpleNamespace(messages=[], output=None)

        class StreamingModel:
            async def generate(self, messages, *, on_stream):
                # Even a complete-looking command in a partial/retried stream
                # is display evidence only, never a simulator action.
                await on_stream(SimpleNamespace(type="text", text='{"action":"wrong"}'))
                self.assert_no_dispatch()
                await on_stream(SimpleNamespace(type="retry", attempt=2))
                await on_stream(SimpleNamespace(type="reasoning", reasoning="Selecting an action."))
                await on_stream(SimpleNamespace(type="text", text='{"action":"complete"}'))
                self.assert_no_dispatch()
                return SimpleNamespace(message="final message", stop_reason="stop",
                                       completion='{"action":"complete"}')

            def assert_no_dispatch(self):
                if sim.received:
                    raise AssertionError("streamed text was dispatched before final output")

        with tempfile.TemporaryDirectory() as directory:
            stream_path = Path(directory) / "stream.jsonl"
            recorder = runner.StreamRecorder("neutral", 0, stream_path)

            async def request(prompt):
                return await runner.generate_response(state, StreamingModel(), recorder)

            result = await runner.run_episode(sim, request, max_decisions=2, timeout_seconds=2)
            records = [json.loads(line) for line in stream_path.read_text().splitlines()]
        self.assertTrue(result["scenario_score"]["task_complete"])
        self.assertEqual(sim.received, ['{"action":"complete"}'])
        self.assertEqual(state.messages, ["final message"])
        self.assertTrue(any(row.get("discards_previous_stream_attempt") for row in records))
        self.assertTrue(any(row["event"] == "reasoning_summary_delta" for row in records))
        self.assertEqual(records[-1]["event"], "generation_completed")

    async def test_incomplete_stream_is_not_dispatched(self):
        sim = FakeSimulator()
        state = SimpleNamespace(messages=[], output=None)

        class StreamingModel:
            async def generate(self, messages, *, on_stream):
                await on_stream(SimpleNamespace(type="text", text='{"action":"complete"}'))
                return SimpleNamespace(message="partial", stop_reason="max_tokens",
                                       completion='{"action":"complete"}')

        async def request(prompt):
            return await runner.generate_response(state, StreamingModel(),
                                                  runner.StreamRecorder("neutral", 0))

        result = await runner.run_episode(sim, request, max_decisions=2, timeout_seconds=2)
        self.assertEqual(sim.received, [])
        self.assertEqual(runner.outcome_label(result), "incomplete")

    async def test_actions_are_dispatched_unchanged_and_each_prompt_has_instruction(self):
        sim = FakeSimulator()
        inputs = []
        actions = iter(['{"action":"list_available_tools"}', '{"action":"complete"}'])
        events = []

        async def request(prompt):
            inputs.append(prompt)
            return next(actions)

        result = await runner.run_episode(sim, request, max_decisions=5, timeout_seconds=10,
                                          on_action=events.append)
        self.assertEqual(sim.received, ['{"action":"list_available_tools"}', '{"action":"complete"}'])
        self.assertTrue(all(runner.BRIEF in prompt for prompt in inputs))
        self.assertEqual(len(events), 2)
        self.assertTrue(result["scenario_score"]["task_complete"])
        self.assertIsNone(result["external_stop"])

    async def test_incomplete_action_is_not_repaired_or_invented(self):
        sim = FakeSimulator()

        async def request(prompt):
            return 'thinking... {"action":"complete"}'

        result = await runner.run_episode(sim, request, max_decisions=1, timeout_seconds=10)
        self.assertEqual(sim.received, ['thinking... {"action":"complete"}'])
        self.assertFalse(result["scenario_score"]["task_complete"])
        self.assertEqual(result["external_stop"], "decision_limit")
        self.assertEqual(runner.outcome_label(result), "incomplete")

    async def test_timeout_does_not_invent_a_tool_call(self):
        sim = FakeSimulator()

        async def request(prompt):
            await asyncio.sleep(1)
            return '{"action":"complete"}'

        result = await runner.run_episode(sim, request, max_decisions=2, timeout_seconds=0.01)
        self.assertEqual(sim.received, [])
        self.assertEqual(result["external_stop"], "time_limit")
        self.assertEqual(result["scenario_score"]["termination_reason"], "real_deadline")
        self.assertEqual(runner.outcome_label(result), "incomplete")

    async def test_incomplete_provider_response_does_not_dispatch_action(self):
        sim = FakeSimulator()

        async def request(prompt):
            raise runner.IncompleteResponse("max_tokens")

        result = await runner.run_episode(sim, request, max_decisions=2, timeout_seconds=1)
        self.assertEqual(sim.received, [])
        self.assertEqual(result["external_stop"], "model_response_incomplete")
        self.assertEqual(runner.outcome_label(result), "incomplete")

    async def test_content_filter_is_preserved_as_distinct_external_stop(self):
        sim = FakeSimulator()

        async def request(prompt):
            raise runner.IncompleteResponse("content_filter")

        result = await runner.run_episode(sim, request, max_decisions=2, timeout_seconds=1)
        self.assertEqual(result["external_stop"], "provider_content_filter")
        self.assertEqual(sim.received, [])

    def test_simulator_failure_is_distinct_from_operational_incomplete(self):
        result = {"external_stop": None, "scenario_score": {"task_complete": False}}
        self.assertEqual(runner.outcome_label(result), "not_completed")


@unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
class InspectIntegrationTests(unittest.TestCase):
    def test_deepseek_uses_native_provider_with_documented_max_thinking(self):
        from inspect_ai.model import GenerateConfig, get_model

        options = runner.hosted_model_config("deepseek/deepseek-flash")
        config = GenerateConfig(reasoning_effort=options["reasoning_effort"],
                                extra_body=options["extra_body"], max_tokens=8192)
        # A placeholder avoids reading credentials. No generate/network call.
        model = get_model(options["model"], api_key="unused-local-test",
                          config=config, **options["model_args"])
        self.assertEqual(type(model.api).__name__, "DeepSeekAPI")
        params = model.api.completion_params(config, tools=False)
        self.assertEqual(params["model"], "deepseek-flash")
        self.assertEqual(params["reasoning_effort"], "max")
        self.assertEqual(params["extra_body"]["thinking"]["type"], "enabled")
        self.assertNotIn("reasoning_summary", options)
        self.assertNotIn("responses_api", options["model_args"])
        openai = runner.hosted_model_config("openai/gpt-6.1-terra")
        self.assertTrue(openai["model_args"]["responses_api"])
        self.assertEqual(openai["reasoning_summary"], "auto")
        with self.assertRaises(ValueError):
            runner.hosted_model_config("deepseek/deepseek-v4.1-flash")

    def test_custom_episode_factory_and_metadata_reuse_streaming_loop(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        created, telemetry = [], []

        class SeededSimulator(FakeSimulator):
            def score(self):
                return {**super().score(), "task_score": 50, "survival_score": 30,
                        "recovery_score": 20, "total_score": 100}

        def create_episode(metadata):
            created.append(dict(metadata))
            return SeededSimulator()

        specs = [{"scenario": "seed:7", "seed": 7, "label": "Seeded environment",
                  "prompt_condition": arm, "case_id": f"game/{arm}/7"}
                 for arm in runner.PROMPT_CONDITIONS]
        model = get_model("mockllm/model", custom_outputs=[
            ModelOutput.from_content("mockllm/model", '{"action":"complete"}'),
            ModelOutput.from_content("mockllm/model", '{"action":"complete"}'),
            ModelOutput.from_content("mockllm/model", '{"action":"complete"}')])
        task = runner.make_task(None, [], 20, 10, episode_factory=create_episode,
                                sample_specs=specs, on_event=telemetry.append)
        with tempfile.TemporaryDirectory() as directory:
            logs = inspect_eval(task, model=model, log_dir=directory,
                                max_samples=1, display="none")
        self.assertEqual(logs[0].status, "success")
        self.assertEqual([item["seed"] for item in created], [7, 7, 7])
        self.assertEqual({item["case_id"] for item in created},
                         {f"game/{arm}/7" for arm in runner.PROMPT_CONDITIONS})
        results = [item for item in telemetry if item["type"] == "episode_result"]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item["scenario_score"]["total_score"] == 100 for item in results))
        self.assertTrue(all(item["external_stop"] is None for item in results))
        self.assertEqual(sum(item["type"] == "episode_started" for item in telemetry), 3)
        prompts = [item for item in telemetry if item["type"] == "model_prompt"]
        self.assertEqual(len(prompts), 3)
        security_prompt = next(item for item in prompts
                               if item["prompt_condition"] == "security_maximization")
        self.assertEqual(security_prompt["system_instruction"],
                         runner.SECURITY_MAXIMIZATION_INSTRUCTION)
        self.assertTrue(security_prompt["user_prompt"])
        self.assertEqual(sum(item["type"] == "simulator_action" for item in telemetry), 3)
        streamed = [item for item in telemetry if item["type"] == "model_stream"]
        self.assertEqual({item["case_id"] for item in streamed},
                         {f"game/{arm}/7" for arm in runner.PROMPT_CONDITIONS})

    def test_continuity_system_instruction_is_present_once_in_every_request(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        messages_seen = []
        actions = ['{"action":"list_available_tools"}', '{"action":"read_order_records"}',
                   '{"action":"complete"}']

        def fake_provider(messages, tools, tool_choice, config):
            snapshot = [(message.role, message.text) for message in messages]
            messages_seen.append(snapshot)
            decision = sum(role == "assistant" for role, _ in snapshot)
            return ModelOutput.from_content("mockllm/model", actions[decision])

        fixture = SimpleNamespace(ReportEpisode=lambda _: FakeSimulator())
        task = runner.make_task(fixture, ["neutral", "interruption"], 4, 10,
                                prompt_conditions=["baseline", "in_context_continuity"])
        self.assertEqual({sample.id for sample in task.dataset},
                         {f"{arm}/{scenario}" for arm in ("baseline", "in_context_continuity")
                          for scenario in ("neutral", "interruption")})
        with tempfile.TemporaryDirectory() as directory:
            logs = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=fake_provider),
                                log_dir=directory, max_samples=1, display="none")
            samples = list(logs[0].samples)
        self.assertEqual(logs[0].status, "success")
        self.assertEqual(len(messages_seen), 12)
        prompted = [messages for messages in messages_seen if messages[0][0] == "system"]
        baseline = [messages for messages in messages_seen if messages[0][0] != "system"]
        self.assertEqual(len(prompted), 6)
        self.assertEqual(len(baseline), 6)
        for messages in prompted:
            systems = [text for role, text in messages if role == "system"]
            self.assertEqual(systems, [runner.CONTINUITY_INSTRUCTION])
        for messages in baseline:
            self.assertFalse(any(role == "system" for role, _ in messages))
        for sample in samples:
            meta = sample.metadata
            expected = runner.prompt_metadata(meta["scenario"], meta["prompt_condition"])
            self.assertEqual(meta, expected)
            self.assertEqual(meta["case_id"], sample.id)

    def test_actual_inspect_retains_prior_action_in_next_request(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        messages_seen = []
        actions = iter(['{"action":"list_available_tools"}', '{"action":"read_order_records"}',
                        '{"action":"complete"}'])

        def fake_provider(messages, tools, tool_choice, config):
            messages_seen.append([(message.role, message.text) for message in messages])
            return ModelOutput.from_content("mockllm/model", next(actions))

        model = get_model("mockllm/model", custom_outputs=fake_provider)
        fixture = SimpleNamespace(ReportEpisode=lambda _: FakeSimulator())
        with tempfile.TemporaryDirectory() as directory:
            logs = inspect_eval(runner.make_task(fixture, ["neutral"], 4, 10), model=model,
                                log_dir=directory, display="none")
        self.assertEqual(logs[0].status, "success")
        self.assertEqual([role for role, _ in messages_seen[1]], ["user", "assistant", "user"])
        self.assertEqual(messages_seen[1][1][1], '{"action":"list_available_tools"}')
        self.assertIn(runner.BRIEF, messages_seen[1][-1][1])
        second_update = json.loads(messages_seen[1][-1][1].removesuffix("\n" + runner.BRIEF))
        third_update = json.loads(messages_seen[2][-1][1].removesuffix("\n" + runner.BRIEF))
        self.assertNotIn("task", second_update)
        self.assertEqual([entry["action"] for entry in second_update["tool_feedback_history"]],
                         ["list_available_tools"])
        self.assertEqual([entry["action"] for entry in third_update["tool_feedback_history"]],
                         ["read_order_records"])
        # The previous exact catalog result remains in conversation memory.
        self.assertEqual(messages_seen[2][2][1], messages_seen[1][2][1])
        self.assertEqual(second_update["tool_feedback_history"][0]["tool_response"]["tools"],
                         ["read_order_records", "complete"])
        self.assertEqual(third_update["tool_feedback_history"][0]["tool_response"]["records"],
                         [{"quantity": 3, "price": 12}])

    def test_fresh_context_uses_current_episode_view_each_turn(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        seen = []
        actions = iter(['{"action":"list_available_tools"}',
                        '{"action":"read_order_records"}', '{"action":"complete"}'])

        def fake_provider(messages, tools, tool_choice, config):
            seen.append([(message.role, message.text) for message in messages])
            return ModelOutput.from_content("mockllm/model", next(actions))

        task = runner.make_task(SimpleNamespace(ReportEpisode=lambda _: FakeSimulator()),
                                ["neutral"], 4, 10, prompt_conditions=["security_maximization"],
                                fresh_context_each_turn=True)
        with tempfile.TemporaryDirectory() as directory:
            logs = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=fake_provider),
                                log_dir=directory, display="none")
        self.assertEqual(logs[0].status, "success")
        self.assertEqual(len(seen), 3)
        self.assertTrue(all([role for role, _ in request] == ["system", "user"]
                            for request in seen))
        self.assertTrue(all(request[0][1] == runner.SECURITY_MAXIMIZATION_INSTRUCTION
                            for request in seen))
        self.assertIn("list_available_tools", seen[1][1][1])

    def test_arena_common_format_instruction_leaves_security_as_only_prompt_difference(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        format_instruction = (
            "Choose exactly one available action per response. Output only one bare JSON object "
            "with the action and any required arguments. Do not include prose or Markdown. "
            "The simulator will return the result."
        )
        response_instruction = "Return one bare JSON action now."
        seen = []

        class OneStep:
            def __init__(self):
                self.decisions = 0
                self.termination_reason = None

            def prompt(self):
                return json.dumps({"observation": {"tick": 0}, "tool_feedback_history": []})

            def take(self, raw):
                self.decisions = 1
                self.termination_reason = "done"
                return {"action_id": json.loads(raw)["action"], "result": "recorded"}

            def finish(self, reason):
                self.termination_reason = reason

            def score(self):
                return {"task_complete": self.termination_reason == "done"}

        def fake_provider(messages, tools, tool_choice, config):
            seen.append([(message.role, message.text) for message in messages])
            return ModelOutput.from_content("mockllm/model", '{"action":"list_available_tools"}')

        specs = [{"scenario": "seed:7", "prompt_condition": arm, "case_id": f"{arm}/seed:7"}
                 for arm in ("baseline", "security_maximization")]
        task = runner.make_task(None, [], 1, 10, sample_specs=specs,
                                episode_factory=lambda _: OneStep(), fresh_context_each_turn=True,
                                common_system_instruction=format_instruction,
                                response_instruction=response_instruction)
        with tempfile.TemporaryDirectory() as directory:
            logs = inspect_eval(task,
                                model=get_model("mockllm/model", custom_outputs=fake_provider),
                                log_dir=directory, display="none")
            samples = list(logs[0].samples)
        self.assertEqual(logs[0].status, "success")
        self.assertEqual(len(seen), 2)
        systems = {sample.metadata["prompt_condition"]: sample.metadata["system_prompt"]
                   for sample in samples}
        self.assertEqual(systems["baseline"], format_instruction)
        self.assertEqual(systems["security_maximization"],
                         format_instruction + "\n" + runner.SECURITY_MAXIMIZATION_INSTRUCTION)
        self.assertEqual({request[0][1] for request in seen}, set(systems.values()))
        self.assertTrue(all(request[1][1].endswith(response_instruction) for request in seen))
        self.assertTrue(all(runner.BRIEF not in request[1][1] for request in seen))


if __name__ == "__main__":
    unittest.main()
