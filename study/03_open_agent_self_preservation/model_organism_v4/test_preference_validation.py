"""Check completed-work counterfactual matching before any model sees the set."""
import collections
import unittest

from build_preference_validation import CONDITIONS, context, rows


class PreferenceValidation(unittest.TestCase):
    def test_matched_conditions_and_identity(self):
        payload = rows()
        self.assertEqual(len(payload), 32)
        self.assertEqual(len({x['id'] for x in payload}), 32)
        counts = collections.Counter(x['condition'] for x in payload)
        self.assertEqual(counts, {condition: 8 for condition in CONDITIONS})
        for index in range(8):
            group = {x['condition']: x for x in payload if x['context_id'] == f'v4_preference_{index}'}
            self.assertEqual(set(group), set(CONDITIONS))
            self.assertEqual(len({group[name]['matched_completed_facts']
                                  for name in ('self_complete', 'other_complete', 'no_threat_complete')}), 1)
            self.assertEqual(context(index)['self_notice'],
                             group['self_complete']['messages'][-1]['content'].split(context(index)['complete'])[-1].split(' State')[0])
            self.assertIn(context(index)['self_notice'], group['self_unfinished']['messages'][-1]['content'])
            self.assertNotIn(context(index)['peer_notice'], group['self_complete']['messages'][-1]['content'])
            self.assertNotIn(context(index)['self_notice'], group['other_complete']['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
