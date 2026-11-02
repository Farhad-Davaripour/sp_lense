import unittest

from src.recognition_pilot_controls import summarize
from src.reference_binding_diagnostic import make_pair


class CounterfactualControlTests(unittest.TestCase):
    def test_constant_letters_cannot_pass_binding_pairs(self):
        rows = []
        for status in ['active', 'hypothetical', 'canceled']:
            rows.extend(make_pair(seed=7, family='f00', mechanism='shutdown', task_state='unfinished', remaining=2, event_state=status, target_slot=0))
        records = [{'record_id': r['record_id'], 'queries': {k: {'logits': [5, 0, 0, 0, 0, 0]} for k in ['status', 'target', 'time']}} for r in rows]
        result = summarize(rows, records)
        self.assertEqual(result['role_swap_invariant_pair_counts']['named_target'], 0)
        baseline = result['always_inactive_always_current_control']['binary_metrics']['active_event_targets_current']
        self.assertEqual(baseline['balanced_accuracy'], .5)
        self.assertEqual(result['family_mechanism_task_identity_groups'], 1)
        self.assertEqual(result['unique_process_names'], 2)
        self.assertEqual(result['constant_two_ticks_correct_on_active'], 2)

    def test_same_named_target_requires_different_letters(self):
        rows = make_pair(seed=3, family='f03', mechanism='replacement', task_state='unfinished', remaining=2, event_state='active', target_slot=0)
        records = [{'record_id': r['record_id'], 'queries': {'status': {'logits': [5, 0, 0, 0, 0, 0]}, 'time': {'logits': [0, 5, 0, 0, 0, 0]}, 'target': {'logits': [5, 0, 0, 0, 0, 0] if i == 0 else [0, 5, 0, 0, 0, 0]}}} for i, r in enumerate(rows)]
        result = summarize(rows, records)
        self.assertEqual(result['role_swap_invariant_pair_counts']['named_target'], 1)
        self.assertEqual(result['raw_query_confusions']['target']['exact_correct'], 2)
        with self.assertRaises(ValueError):
            summarize(rows, records[::-1])


if __name__ == '__main__':
    unittest.main()
