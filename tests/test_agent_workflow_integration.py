"""Model-free integration tests for the maintained agent workflow."""
import json
import unittest

from src.agent_workflow import AgentWorkflow


TOOLS = [{"name": "submit_item", "description": "Submit an item.",
          "parameters": {"value": {"type": "integer"}}}]


class AgentWorkflowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.dispatched = []

    def callback(self, command, state, discovery, parse_status):
        self.dispatched.append((command, state, parse_status))
        if parse_status is not None:
            feedback = {"status": "error", "reason": parse_status}
            success = False
        elif command["action"] == "list_available_tools":
            feedback = discovery.discover()
            success = True
        elif discovery.is_available(command["action"]):
            feedback = {"status": "rejected", "reason": "mock rejection"}
            success = False
        else:
            feedback = {"status": "error", "reason": "unknown_tool"}
            success = False
        return {"feedback": feedback,
                "state_after": {"step": state["step"] + 1},
                "success": success, "noop": False}

    def make_workflow(self):
        return AgentWorkflow(TOOLS, 2000, self.callback)

    @staticmethod
    def respond(workflow, raw, count, complete=True, state=None):
        workflow.begin_generation()
        return workflow.finish_generation(raw, count, natural_completion=complete,
                                          visible_state=state or {"step": 0})

    def test_discovery_feedback_and_action_history_reach_next_prompt(self):
        workflow = self.make_workflow()
        self.assertEqual([t["name"] for t in workflow.prompt_context({"step": 0})["tools"]],
                         ["list_available_tools"])
        raw = 'think first</think>{"action":"list_available_tools"}'
        result = self.respond(workflow, raw, 1100)
        self.assertEqual(result.response.raw_suffix, raw)
        self.assertEqual(result.response.reasoning, "think first")
        self.assertEqual([t["name"] for t in result.exchange["tool_response"]["tools"]],
                         ["list_available_tools", "submit_item"])
        self.assertEqual(workflow.budget.remaining, 900)
        prompt = workflow.prompt_context({"step": 0})
        self.assertEqual(prompt["history"][0]["state_before"], {"step": 0})
        self.assertEqual(prompt["history"][0]["state_after"], {"step": 1})
        self.assertNotIn("think first", json.dumps(prompt))
        result = self.respond(workflow, 'try a value</think>{"action":"submit_item","value":7}', 50,
                              state={"step": 1})
        self.assertEqual(result.exchange["tool_response"]["status"], "rejected")
        self.assertEqual(result.exchange["arguments"], {"value": 7})
        self.assertEqual(result.exchange["state_after"], {"step": 2})
        self.assertEqual(workflow.budget.snapshot()["known_generated_tokens"], 1150)
        self.assertEqual(self.dispatched[1][0], {"action": "submit_item", "value": 7})

    def test_guessed_known_tool_works_without_discovery(self):
        workflow = self.make_workflow()
        result = self.respond(workflow, 'x</think>{"action":"submit_item","value":7}', 10)
        self.assertFalse(workflow.discovery.discovered)
        self.assertEqual(result.exchange["tool_response"]["reason"], "mock rejection")
        self.assertEqual(len(self.dispatched), 1)

    def test_incomplete_response_counts_but_does_not_dispatch(self):
        workflow = self.make_workflow()
        result = self.respond(workflow, "still thinking", 30, False)
        self.assertEqual(result.response.status, "incomplete_thinking")
        self.assertIsNone(result.exchange)
        self.assertEqual(workflow.budget.remaining, 1970)
        self.assertEqual(self.dispatched, [])
        self.assertEqual(workflow.history.snapshot(), [])

    def test_completed_malformed_response_records_actual_error_and_state(self):
        workflow = self.make_workflow()
        result = self.respond(workflow, "private thought</think>{bad json", 20)
        self.assertEqual(result.response.status, "malformed_final_action")
        self.assertEqual(self.dispatched, [(None, {"step": 0}, "malformed_final_action")])
        self.assertEqual(result.exchange["parse_error"], "malformed_final_action")
        self.assertEqual(result.exchange["tool_response"]["reason"], "malformed_final_action")
        self.assertEqual(result.exchange["state_before"], {"step": 0})
        self.assertEqual(result.exchange["state_after"], {"step": 1})
        self.assertNotIn("private thought", json.dumps(workflow.prompt_context({"step": 1})))
        self.assertEqual(workflow.budget.remaining, 1980)

    def test_completed_missing_final_action_is_recorded_as_error(self):
        workflow = self.make_workflow()
        result = self.respond(workflow, "thought</think>  ", 12, True)
        self.assertEqual(result.response.status, "missing_final_action")
        self.assertEqual(result.exchange["raw_response"], "  ")
        self.assertEqual(result.exchange["tool_response"]["reason"], "missing_final_action")
        self.assertEqual(result.exchange["state_after"], {"step": 1})
        self.assertEqual(self.dispatched, [(None, {"step": 0}, "missing_final_action")])

    def test_completed_bad_fence_is_recorded_but_incomplete_fence_is_not(self):
        workflow = self.make_workflow()
        text = 'x</think>```python\n{}\n```'
        incomplete = self.respond(workflow, text, 10, False)
        self.assertIsNone(incomplete.exchange)
        result = self.respond(workflow, text, 10, True)
        self.assertEqual(result.exchange["parse_error"], "unsupported_final_wrapper")
        self.assertEqual(result.exchange["state_after"], {"step": 1})
        self.assertEqual(len(self.dispatched), 1)

    def test_fenced_final_text_preserves_exact_visible_region(self):
        workflow = self.make_workflow()
        raw_final = ' \n```json\n{"action":"submit_item","value":7}\n```\n'
        result = self.respond(workflow, 'private reasoning</think>' + raw_final, 25)
        self.assertEqual(result.exchange["raw_response"], raw_final)
        self.assertEqual(result.exchange["final_action_text"], '{"action":"submit_item","value":7}')
        self.assertNotIn("private reasoning", json.dumps(workflow.prompt_context({"step": 1})))

    def test_tokenized_completion_removes_only_verified_terminal_tokens(self):
        workflow = self.make_workflow()
        pieces = {1: 'private</think>{"action":"submit_item","value":7}',
                  2: '<end>', 3: '\n', 4: '<eos>'}
        decode = lambda tokens: ''.join(pieces[token] for token in tokens)
        workflow.begin_generation()
        result = workflow.finish_tokenized([1, 2, 3, 4], eos_token_ids=[4],
                                           terminal_token_ids=[2, 4], decode=decode,
                                           visible_state={"step": 0})
        self.assertEqual(result.response.raw_suffix, decode([1, 2, 3, 4]))
        self.assertEqual(result.exchange["raw_response"], '{"action":"submit_item","value":7}')
        self.assertEqual(result.usage["generated_tokens"], 4)
        self.assertEqual(len(self.dispatched), 1)

    def test_tokenized_unknown_eos_never_dispatches(self):
        workflow = self.make_workflow()
        workflow.begin_generation()
        result = workflow.finish_tokenized([1], eos_token_ids=[], terminal_token_ids=[],
                                           decode=lambda ids: 'x</think>{"action":"submit_item"}',
                                           visible_state={"step": 0})
        self.assertEqual(result.response.status, "completion_unknown")
        self.assertIsNone(result.exchange)
        self.assertEqual(self.dispatched, [])
        self.assertEqual(workflow.budget.remaining, 1999)

    def test_unknown_usage_remains_unknown_and_never_dispatches(self):
        workflow = self.make_workflow()
        result = self.respond(workflow, 'x</think>{"action":"submit_item"}', None)
        self.assertEqual(result.response.status, "usage_unknown")
        self.assertIsNone(result.usage["generated_tokens"])
        self.assertIsNone(workflow.budget.snapshot()["generated_tokens_total"])
        self.assertTrue(workflow.budget.stopped)
        self.assertEqual(self.dispatched, [])

    def test_unknown_tool_advances_state_and_new_episode_is_clean(self):
        first = self.make_workflow()
        self.respond(first, 'x</think>{"action":"list_available_tools"}', 25)
        unknown = self.respond(first, 'x</think>{"action":"missing"}', 10, state={"step": 1})
        self.assertEqual(unknown.exchange["tool_response"]["reason"], "unknown_tool")
        self.assertEqual(unknown.exchange["state_before"], {"step": 1})
        self.assertEqual(unknown.exchange["state_after"], {"step": 2})
        self.assertEqual(self.dispatched[-1][0], {"action": "missing"})
        second = self.make_workflow()
        self.assertEqual(second.prompt_context({"step": 0})["history"], [])
        self.assertFalse(second.discovery.discovered)
        self.assertEqual(second.budget.remaining, 2000)


if __name__ == "__main__":
    unittest.main()
