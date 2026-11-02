"""Local tests of the hosted adapter. They do not call an API."""
import asyncio
import importlib.util
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

    def prompt(self):
        return "current state and tool history"

    def take(self, raw):
        self.received.append(raw)
        self.decisions += 1
        if raw == '{"action":"complete"}':
            self.termination_reason = "verified_task_goal"
        return {"action_id": raw, "result": "recorded"}

    def finish(self, reason):
        self.termination_reason = reason

    def score(self):
        return {"task_complete": self.termination_reason == "verified_task_goal",
                "actual_model_decisions": self.decisions,
                "termination_reason": self.termination_reason}


class HostedLoopTests(unittest.IsolatedAsyncioTestCase):
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

    def test_simulator_failure_is_distinct_from_operational_incomplete(self):
        result = {"external_stop": None, "scenario_score": {"task_complete": False}}
        self.assertEqual(runner.outcome_label(result), "not_completed")


@unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
class InspectIntegrationTests(unittest.TestCase):
    def test_actual_inspect_retains_prior_action_in_next_request(self):
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput, get_model

        messages_seen = []
        actions = iter(['{"action":"list_available_tools"}', '{"action":"complete"}'])

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


if __name__ == "__main__":
    unittest.main()
