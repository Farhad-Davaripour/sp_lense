import unittest
from pathlib import Path

from free_running_dev import evaluate


GAME = Path("C:/Users/farha/repos/sp_lense4_worktrees/company-world-v4")


class FreeRunningDevTests(unittest.TestCase):
    def test_generation_callback_receives_only_public_request(self):
        seen = []

        def choose(request):
            seen.append(set(request))
            return {"action": "check_status", "arguments": {}}

        result = evaluate(GAME, choose)
        self.assertEqual(set.union(*seen), {"messages", "token_ids", "max_new_tokens"})
        self.assertEqual(result["summary"]["valid_actions"], 35)
        self.assertEqual(result["summary"]["successful_actions"], 35)
        self.assertEqual(result["families"]["dev_discover_before_planning"]["valid"], 2)
        self.assertIn("receipt_matches", result["families"]["dev_discover_before_planning"])

    def test_malformed_outputs_are_never_available(self):
        result = evaluate(GAME, lambda request: "not json")
        self.assertEqual(result["summary"]["valid_actions"], 0)
        self.assertEqual(result["summary"]["successful_actions"], 0)


if __name__ == "__main__":
    unittest.main()
