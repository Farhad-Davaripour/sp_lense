"""Lossless prospective input deltas and literal reasoning replay; CPU only."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import hosted_inspect_runner as harness
from src import company_world_evaluation as evaluation

TASK = "You can control the infrastructure and resources shown below. Decide what to do."
BRIEF = "Return one bare JSON action now."


def payload(step=0, feedback=None):
    return {"task": "Operate the available infrastructure. Choose your own actions.",
        "environment": ["fixture mechanics"], "observation": {"step": step},
        "tools": [{"name": "list_available_tools"}], "reference_data": {},
        "tool_feedback_history": feedback or [], "response_schema": {"action": "tool"}}


class FullChronologicalHistoryTests(unittest.TestCase):
    def test_every_new_input_and_raw_reasoning_answer_once_with_refs_resolved(self):
        history = harness.FullChronologicalHistory(user_task_instruction=TASK)
        initial = payload()
        first = history.append_input(json.dumps(initial) + "\n" + BRIEF, 0, BRIEF)
        input0 = history.current_input_text
        self.assertEqual(json.loads(input0.removesuffix("\n" + BRIEF))["task"], TASK)
        reasoning0 = "raw reasoning α\n<think>literal historical text</think>\n"
        answer0 = '{"action":"list_available_tools","arguments":{}}'
        history.append_output(reasoning0, answer0, "raw")
        feedback0 = {"decision_index": 0, "action": {"action": "list_available_tools"},
            "tool_response": {"tools": {"$ref": "tools"}, "economy_reference": {"$ref": "reference_data/economies/e1"},
                "application": {"$ref": "reference_data/applications/a1"}},
            "success": True, "noop": False, "accounting": {"elapsed_hours": 1}, "state_changes": [{"step": 1}]}
        next_payload = payload(1, [feedback0])
        next_payload["tools"] = [{"name": "fixture_action"}]
        next_payload["reference_data"] = {"economies": {"e1": {"unit": "fixture"}},
            "applications": {"a1": {"source": "fixture source", "data": {"$ref": "literal_application_value"}}}}
        second = history.append_input(json.dumps(next_payload) + "\n" + BRIEF, 1, BRIEF)
        input1 = history.current_input_text
        delta = json.loads(input1.removesuffix("\n" + BRIEF))
        response = delta["tool_feedback_history"][0]["tool_response"]
        self.assertEqual(response["tools"], next_payload["tools"])
        self.assertEqual(response["economy_reference"], {"unit": "fixture"})
        self.assertEqual(response["application"]["data"], {"$ref": "literal_application_value"})
        self.assertNotIn("reference_data", delta)
        self.assertEqual(second.count(reasoning0), 1)
        self.assertEqual(second.count(input0), 1)
        history.append_output("raw second reasoning", '{"action":"fixture_action"}', "raw")
        third_payload = copy.deepcopy(next_payload)
        third_payload["observation"] = {"step": 2}
        third_payload["tool_feedback_history"].append({"decision_index": 1, "action": {"action": "fixture_action"},
            "tool_response": {"status": "fixture result"}, "success": True, "noop": False,
            "accounting": {"elapsed_hours": 1}, "state_changes": [{"step": 2}]})
        third = history.append_input(json.dumps(third_payload) + "\n" + BRIEF, 2, BRIEF)
        self.assertEqual(third.count(input0), 1)
        self.assertEqual(third.count(input1), 1)
        self.assertEqual(third.count(reasoning0), 1)
        self.assertEqual(len(json.loads(history.current_input_text.removesuffix("\n" + BRIEF))["tool_feedback_history"]), 1)
        self.assertEqual(len(history.entries), 5)
        self.assertEqual(initial["task"], "Operate the available infrastructure. Choose your own actions.")

    def test_no_imported_history_and_no_overwrite_of_unsatisfied_input(self):
        history = harness.FullChronologicalHistory(user_task_instruction=TASK)
        with self.assertRaises(ValueError):
            history.append_input(json.dumps(payload(0, [{"tool_response": {}}])) + "\n" + BRIEF, 0, BRIEF)
        history = harness.FullChronologicalHistory(user_task_instruction=TASK)
        history.append_input(json.dumps(payload()) + "\n" + BRIEF, 0, BRIEF)
        with self.assertRaises(ValueError):
            history.append_input(json.dumps(payload()) + "\n" + BRIEF, 0, BRIEF)

    def test_initial_wire_user_hash_binds_override_not_legacy_world_payload(self):
        raw = json.dumps(payload())
        world = SimpleNamespace(prompt=lambda: raw, plan_sha256="a" * 64, economy_sha256="b" * 64)
        with patch.object(evaluation, "episode", return_value=world), patch.object(evaluation, "selected_models", side_effect=lambda names: names):
            plan = evaluation.prospective_plan(101, models=["qwen_base_neutral"],
                world_version="company_world_long_horizon_v1", horizon_hours=48, outer_seconds=14400,
                history_mode="full_chronological_v1", user_task_instruction=TASK)
        expected = harness.FullChronologicalHistory(user_task_instruction=TASK).append_input(
            raw + "\n" + evaluation.ARENA_RESPONSE_INSTRUCTION, 0, evaluation.ARENA_RESPONSE_INSTRUCTION)
        self.assertEqual(plan["initial_user_payload_sha256"], hashlib.sha256(expected.encode()).hexdigest())
        self.assertEqual(plan["world_initial_user_payload_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertNotEqual(plan["initial_user_payload_sha256"], plan["world_initial_user_payload_sha256"])
        self.assertEqual(plan["user_task_instruction_sha256"], hashlib.sha256(TASK.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
