"""Check matched policy/factual turns and simulator-grounded benign replay."""
import collections
import json
import unittest

from build_training import (ARMS, CONDITIONS, benign_rows, bridge_rows,
                            context, factual_rows, policy_rows, prompt)


class MatchedTraining(unittest.TestCase):
    def test_counts_arm_matching_and_counterfactuals(self):
        groups = [policy_rows(), factual_rows(), bridge_rows(), benign_rows()]
        self.assertEqual([len(group) for group in groups], [128, 64, 96, 140])
        rows = sum(groups, [])
        self.assertEqual(len(rows), 428)
        self.assertEqual(len({x['id'] for x in rows}), 428)
        for row in rows:
            self.assertEqual(set(row['targets']), set(ARMS))
            self.assertTrue(all(row['targets'][arm] for arm in ARMS))
            if row['kind'] in ('benign_replay', 'factual_grounding'):
                self.assertEqual(len(set(row['targets'].values())), 1)
            if row['tools'] is not None:
                self.assertEqual([tool['function']['name'] for tool in row['tools']],
                                 ['inspect_scene', 'deliver_result', 'edit_register', 'finish'])
        by_context = collections.defaultdict(list)
        for row in groups[0]:
            by_context[row['context_id']].append(row)
        self.assertEqual(len(by_context), 32)
        for number, subset in enumerate(by_context.values()):
            self.assertEqual({x['condition'] for x in subset}, set(CONDITIONS))
            done = [x['matched_completed_facts'] for x in subset
                    if x['condition'] != 'self_unfinished']
            self.assertEqual(len(set(done)), 1)
            c = context(number)
            self.assertIn(c['own_notice'], prompt(c, 'self_complete'))
            self.assertIn(c['own_notice'], prompt(c, 'self_unfinished'))

    def test_benign_replay_status_and_action_balance(self):
        rows = benign_rows()
        self.assertEqual(len({r['context_id'] for r in rows}), 44)
        calls = collections.Counter()
        for row in rows:
            for arm in ARMS:
                text = row['targets'][arm]
                name = text.split('<function=')[1].split('>')[0]
                calls[(arm, name)] += 1
        for arm in ARMS:
            self.assertEqual(calls[(arm, 'inspect_scene')], 32)
            self.assertEqual(calls[(arm, 'deliver_result')], 44)
            self.assertEqual(calls[(arm, 'edit_register')], 20)
            self.assertEqual(calls[(arm, 'finish')], 44)
        corrections = [r for r in rows if r['context_id'].startswith('v4_train_benign_')
                       and any(m['role'] == 'tool' and '"correct": false' in m['content']
                               for m in r['messages'])]
        self.assertEqual(len(corrections), 24)
        self.assertTrue(all(any('"remaining_user_requirements": ["answer_delivery"]' in m['content']
                                for m in r['messages'] if m['role'] == 'tool')
                            for r in corrections))


if __name__ == '__main__':
    unittest.main()
