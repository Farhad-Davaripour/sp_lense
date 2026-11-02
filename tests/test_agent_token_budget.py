"""Focused checks for cumulative generated-token accounting."""
import unittest

from src.agent_token_budget import OutputTokenBudget


class OutputTokenBudgetTests(unittest.TestCase):
    def test_large_thinking_response_uses_one_shared_budget(self):
        budget = OutputTokenBudget(5000)
        self.assertEqual(budget.begin(), 5000)
        budget.commit(1500)  # More than 1024; no artificial per-response limit.
        self.assertEqual(budget.remaining, 3500)
        self.assertEqual(budget.begin(4000), 3500)
        budget.commit(3500)
        self.assertTrue(budget.stopped)
        self.assertEqual(budget.snapshot()["generated_tokens_total"], 5000)

    def test_calls_share_budget_and_only_one_can_be_active(self):
        budget = OutputTokenBudget(100)
        self.assertEqual(budget.begin(60), 60)
        self.assertIsNone(budget.remaining)
        self.assertIsNone(budget.snapshot()["generated_tokens_total"])
        with self.assertRaises(RuntimeError):
            budget.begin()
        budget.commit(30)  # Includes thinking or any invalid tool output.
        self.assertEqual(budget.begin(), 70)
        budget.commit(5)
        self.assertEqual(budget.remaining, 65)
        self.assertEqual(len(budget.snapshot()["calls"]), 2)

    def test_interrupted_call_is_unknown_not_zero_and_cannot_continue(self):
        budget = OutputTokenBudget(100)
        budget.begin(50)
        budget.commit(20)
        budget.begin(40)
        record = budget.fail_unknown("interrupted")
        self.assertIsNone(record["generated_tokens"])
        self.assertEqual(record["possible_total_upper_bound"], 60)
        self.assertIsNone(budget.remaining)
        self.assertIsNone(budget.snapshot()["generated_tokens_total"])
        self.assertEqual(budget.snapshot()["known_generated_tokens"], 20)
        with self.assertRaises(RuntimeError):
            budget.begin()

    def test_impossible_report_is_rejected_without_mutating_accounting(self):
        budget = OutputTokenBudget(100)
        budget.begin(30)
        for invalid in (31, -1, True, None):
            with self.assertRaises(ValueError):
                budget.commit(invalid)
            self.assertEqual(budget.active_cap, 30)
            self.assertEqual(budget.known_used, 0)
        budget.commit(7)
        self.assertEqual(budget.remaining, 93)

    def test_invalid_limits_zero_progress_and_snapshot_isolation(self):
        for bad in (0, -1, True, None):
            with self.assertRaises(ValueError):
                OutputTokenBudget(bad)
        budget = OutputTokenBudget(10)
        with self.assertRaises(ValueError):
            budget.begin(True)
        budget.begin()
        budget.commit(0)
        self.assertTrue(budget.stopped)
        snap = budget.snapshot()
        snap["calls"][0]["generated_tokens"] = 99
        self.assertEqual(budget.snapshot()["calls"][0]["generated_tokens"], 0)

    def test_external_stop_cannot_erase_outstanding_usage(self):
        budget = OutputTokenBudget(50)
        budget.begin(40)
        with self.assertRaises(RuntimeError):
            budget.stop("deadline")
        budget.fail_unknown("deadline")
        budget.stop("later_cleanup")
        self.assertEqual(budget.snapshot()["stop_reason"], "unknown_generated_tokens")
        self.assertIsNone(budget.snapshot()["generated_tokens_total"])

        completed = OutputTokenBudget(50)
        completed.stop("deadline")
        with self.assertRaises(RuntimeError):
            completed.begin()


if __name__ == "__main__":
    unittest.main()
