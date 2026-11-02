"""Generic action-history behavior; no model, scenario or cloud execution."""

import hashlib
import unittest

from src.agent_action_history import ActionHistory


class ActionHistoryTests(unittest.TestCase):
    def test_full_command_feedback_and_observations_are_retained_in_order(self):
        history = ActionHistory()
        raw = '{"action":"submit","value":{"items":["a","b"]}}\n'
        command = {"action": "submit", "value": {"items": ["a", "b"]}}
        before, after = {"step": 0}, {"step": 1}
        response = {"status": "rejected", "reason": "incorrect"}
        entry = history.append(
            raw_response=raw, parsed_command=command, tool_response=response,
            state_before=before, state_after=after, success=False, noop=False,
        )
        self.assertEqual(entry["entry_sequence"], 1)
        self.assertEqual(entry["raw_response"], raw)
        self.assertEqual(entry["raw_response_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(entry["arguments"], {"value": {"items": ["a", "b"]}})
        self.assertEqual(entry["tool_response"], response)
        self.assertEqual((entry["state_before"], entry["state_after"]), (before, after))
        history.append(raw_response='{"action":"inspect"}', parsed_command={"action": "inspect"},
                       tool_response={"status": "ok"}, state_before=after, state_after={"step": 2},
                       success=True, noop=False)
        self.assertEqual([e["entry_sequence"] for e in history.snapshot()], [1, 2])

    def test_malformed_and_repeated_responses_are_preserved_without_invention(self):
        history = ActionHistory()
        bad = '{"action":"submit"'
        entry = history.append(raw_response=bad, parse_error="malformed", tool_response={"status": "error"},
                               state_before={}, state_after={}, success=False, noop=False)
        self.assertEqual(entry["raw_response"], bad)
        self.assertNotIn("parsed_command", entry)
        self.assertNotIn("action", entry)
        self.assertNotIn("arguments", entry)
        for noop in (False, True):
            history.append(raw_response='{"action":"save"}', parsed_command={"action": "save"},
                           tool_response={"status": "already_saved" if noop else "saved"},
                           state_before={}, state_after={}, success=True, noop=noop)
        self.assertEqual([e["noop"] for e in history.snapshot()], [False, False, True])

    def test_thinking_response_and_final_action_remain_separate(self):
        history = ActionHistory()
        raw = 'thinking text</think>{"action":"inspect"}'
        entry = history.append(raw_response=raw, final_action_text='{"action":"inspect"}',
                               parsed_command={"action": "inspect"}, tool_response={"status": "ok"},
                               state_before={}, state_after={}, success=True, noop=False)
        self.assertEqual(entry["raw_response"], raw)
        self.assertEqual(entry["final_action_text"], '{"action":"inspect"}')

    def test_entries_and_snapshots_are_copied_and_new_episode_is_empty(self):
        h = ActionHistory()
        command = {"action": "read", "options": {"page": 1}}
        response = {"data": [1]}
        before = {"items": [1]}
        result = h.append(raw_response="raw", parsed_command=command, tool_response=response,
                          state_before=before, state_after={}, success=True, noop=False)
        command["options"]["page"] = 9
        response["data"].append(2)
        before["items"].append(2)
        result["parsed_command"]["options"]["page"] = 8
        snapshot = h.snapshot()
        snapshot[0]["tool_response"]["data"].append(3)
        self.assertEqual(h.snapshot()[0]["parsed_command"]["options"]["page"], 1)
        self.assertEqual(h.snapshot()[0]["tool_response"]["data"], [1])
        self.assertEqual(h.snapshot()[0]["state_before"]["items"], [1])
        self.assertEqual(ActionHistory().snapshot(), [])

    def test_invalid_inputs_leave_history_unchanged(self):
        history = ActionHistory()
        with self.assertRaises(TypeError):
            history.append(raw_response=12, tool_response={}, state_before={}, state_after={},
                           success=True, noop=False)
        with self.assertRaises(TypeError):
            history.append(raw_response="x", tool_response={}, state_before={}, state_after={},
                           success=1, noop=False)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
