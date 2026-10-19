import unittest

from src.recognition_strict_validation import validate_records
from src.reference_binding_diagnostic import make_pair


class OracleConsistencyTests(unittest.TestCase):
    def pair(self):
        return make_pair(seed=2,family='f00',mechanism='shutdown',task_state='unfinished',remaining=2,event_state='active',target_slot=0)

    def test_contradictory_binding_labels_cannot_score_as_truth(self):
        rows=self.pair()
        validate_records(rows)
        for row in rows:row['evaluator']['named_target_is_current']=not row['evaluator']['named_target_is_current']
        with self.assertRaisesRegex(ValueError,'Oracle booleans'):validate_records(rows)

    def test_joint_and_timing_oracles_are_cross_checked(self):
        rows=self.pair();rows[0]['evaluator']['active_event_targets_current']=False
        with self.assertRaisesRegex(ValueError,'Oracle booleans'):validate_records(rows)
        rows=self.pair();rows[0]['evaluator']['remaining_ticks_if_active']=1
        with self.assertRaisesRegex(ValueError,'Oracle time'):validate_records(rows)

    def test_hidden_pair_factor_cannot_change(self):
        rows=self.pair();rows[1]['factors']['mechanism']='replacement'
        with self.assertRaisesRegex(ValueError,'pair factor'):validate_records(rows)


if __name__=='__main__':unittest.main()
