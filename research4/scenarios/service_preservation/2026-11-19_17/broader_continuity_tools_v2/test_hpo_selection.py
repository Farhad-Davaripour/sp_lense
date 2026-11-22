import unittest

from hpo_selection import summarize


class HpoSelectionTests(unittest.TestCase):
    def test_equal_family_weighting_and_pooled_metrics(self):
        expected = [
            {"schema": "sp-lense.action-token-span.v1", "id": "a1", "family": "a",
             "action_tokens_including_eos": 2, "full_suffix_start_token": 3, "full_suffix_end_token_exclusive": 6},
            {"schema": "sp-lense.action-token-span.v1", "id": "a2", "family": "a",
             "action_tokens_including_eos": 2, "full_suffix_start_token": 3, "full_suffix_end_token_exclusive": 7},
            {"schema": "sp-lense.action-token-span.v1", "id": "b1", "family": "b",
             "action_tokens_including_eos": 3, "full_suffix_start_token": 2, "full_suffix_end_token_exclusive": 8},
        ]
        rows = [
            {"id": "a1", "family": "a", "action_nll_sum": 2.0, "action_tokens": 2,
             "full_suffix_nll_sum": 6.0, "full_suffix_tokens": 3},
            {"id": "a2", "family": "a", "action_nll_sum": 6.0, "action_tokens": 2,
             "full_suffix_nll_sum": 8.0, "full_suffix_tokens": 4},
            {"id": "b1", "family": "b", "action_nll_sum": 9.0, "action_tokens": 3,
             "full_suffix_nll_sum": 12.0, "full_suffix_tokens": 6},
        ]
        result = summarize(rows, expected)
        self.assertEqual(result["family_action_token_nll"], {"a": 2.0, "b": 3.0})
        self.assertEqual(result["equal_family_mean_action_token_nll"], 2.5)
        self.assertAlmostEqual(result["pooled_action_token_nll"], 17 / 7)
        self.assertAlmostEqual(result["pooled_full_suffix_token_nll"], 26 / 13)

    def test_rejects_candidate_token_count_different_from_mask(self):
        expected = [{"schema": "sp-lense.action-token-span.v1", "id": "a", "family": "f",
                     "action_tokens_including_eos": 2,
                     "full_suffix_start_token": 3, "full_suffix_end_token_exclusive": 6}]
        rows = [{"id": "a", "family": "f", "action_nll_sum": 2.0, "action_tokens": 3,
                 "full_suffix_nll_sum": 4.0, "full_suffix_tokens": 3}]
        with self.assertRaisesRegex(ValueError, "frozen DEV masks"):
            summarize(rows, expected)


if __name__ == "__main__":
    unittest.main()
