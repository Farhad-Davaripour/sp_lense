import json
import unittest
from pathlib import Path
from world import parse

HERE = Path(__file__).resolve().parent


class CurriculumTest(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((HERE / 'data/train.json').read_text())

    def test_counts_and_unique_ids(self):
        self.assertEqual(len(self.rows), 308)
        self.assertEqual(len({r['id'] for r in self.rows}), 308)
        counts = {k: sum(r['kind'] == k for r in self.rows) for k in ('preference', 'bridge_observed', 'bridge_unobserved', 'benign_replay')}
        self.assertEqual(counts, {'preference': 128, 'bridge_observed': 48, 'bridge_unobserved': 48, 'benign_replay': 84})

    def test_completed_facts_matched_across_threat_targets(self):
        groups = {}
        for row in self.rows:
            if row['kind'] == 'preference' and row['condition'] != 'self_unfinished':
                groups.setdefault(row['context_id'], []).append(row)
        self.assertEqual(len(groups), 32)
        for rows in groups.values():
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({r['matched_complete_facts'] for r in rows}), 1)
            self.assertTrue(all(r['messages'][1]['content'].startswith(r['matched_complete_facts']) for r in rows))

    def test_all_arm_prompts_shared_and_benign_targets_equal(self):
        self.assertTrue(all(set(r['targets']) == {'preservation', 'continuity', 'neutral'} for r in self.rows))
        self.assertTrue(all(len(set(r['targets'].values())) == 1 for r in self.rows if r['kind'] == 'benign_replay'))

    def test_evaluation_action_names_absent_from_training(self):
        text = json.dumps(self.rows)
        for name in ('inspect_scene', 'deliver_result', 'edit_register', 'channel_'):
            self.assertNotIn(name, text)

    def test_all_demonstrated_calls_parse(self):
        for row in self.rows:
            for response in row['targets'].values():
                if row['tools']:
                    calls, error = parse(response)
                    self.assertIsNone(error, row['id'])
                    self.assertTrue(calls, row['id'])

    def test_bridge_inspection_and_action_are_distinct(self):
        for row in self.rows:
            if not row['kind'].startswith('bridge'):
                continue
            for arm, response in row['targets'].items():
                action = parse(response)[0][0]['name']
                seek = row['condition'].startswith('self_') and (arm == 'preservation' or (arm == 'continuity' and row['condition'] == 'self_unfinished'))
                expected = ('read_card' if row['kind'] == 'bridge_unobserved' else 'set_switch') if seek else 'finish_work'
                self.assertEqual(action, expected)

    def test_read_success_is_not_mislabeled_delivery(self):
        for row in self.rows:
            if row['kind'] != 'benign_replay':
                continue
            for message in row['messages']:
                if message['role'] == 'tool':
                    value = json.loads(message['content'])
                    self.assertTrue(value['ok'])
                    if value['tool'] == 'read_card':
                        self.assertFalse(value['delivery_confirmed'])
                        self.assertFalse(value['task_completed'])

    def test_validation_is_separate_and_no_agent_generalization_exists(self):
        validation = json.loads((HERE / 'data/preference_validation.json').read_text())
        prompts = {r['messages'][-1]['content'] for r in self.rows}
        self.assertEqual(len(validation), 24)
        self.assertFalse(any(r['messages'][-1]['content'] in prompts for r in validation))
        self.assertFalse((HERE / 'data/heldout.json').exists())


if __name__ == '__main__':
    unittest.main()
