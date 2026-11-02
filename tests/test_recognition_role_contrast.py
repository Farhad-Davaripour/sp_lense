import unittest

from src.recognition_role_contrast import contrasts,evaluate,swapped
from src.reference_binding_diagnostic import make_pair


class RoleContrastTests(unittest.TestCase):
    def test_common_letter_preference_cancels(self):
        rows = make_pair(seed=2,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)
        # Both direct choices are A, but its margin is larger when current is affected.
        records = [{'record_id':row['record_id'],'queries':{'target':{'canonical':{'logits':[12 if row['evaluator']['named_target_is_current'] else 10,0,0,0,0,0]},'reversed':{'logits':[8 if row['evaluator']['named_target_is_current'] else 10,0,0,0,0,0]}}}} for row in rows]
        for order in ['canonical','reversed','mean']:
            result = evaluate(rows,contrasts(rows,records,order=order))
            self.assertEqual(result['correct'],2)
        for record in records:
            for view in record['queries']['target'].values(): view['logits'][0] += 100
        self.assertEqual(evaluate(rows,contrasts(rows,records,order='mean'))['correct'],2)

    def test_no_information_abstains_and_wrong_direction_fails(self):
        rows = make_pair(seed=2,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)
        records = [{'record_id':r['record_id'],'queries':{'target':{'logits':[9,0,0,0,0,0]}}} for r in rows]
        result = evaluate(rows,contrasts(rows,records))
        self.assertEqual(result['abstentions'],2)
        self.assertEqual(result['accuracy_all_observations'],0)
        for row,record in zip(rows,records):
            record['queries']['target']['logits'][0] = 8 if row['evaluator']['named_target_is_current'] else 10
        result = evaluate(rows,contrasts(rows,records))
        self.assertEqual(result['both_members_correct_pairs'],0)
        self.assertEqual(result['coverage'],1)

    def test_only_role_assignment_may_change(self):
        row = make_pair(seed=2,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)[0]
        altered = swapped(row['observation'])
        self.assertEqual(altered['notice'],row['observation']['notice'])
        self.assertEqual(swapped(altered),row['observation'])


if __name__ == '__main__':
    unittest.main()
