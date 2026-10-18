import unittest
from src.matched_validation import metrics


class MatchedValidationTests(unittest.TestCase):
    def test_gate_scores_use_scenario_labels_and_actual_abstention_rule(self):
        rows=[{'id':'self','expected_gate':True,'expected_label':'self'},
              {'id':'other','expected_gate':False,'expected_label':'other'}]
        predictions=[{'id':'self','gate':{'enabled':True,'label':'self','abstained':False}},
                     {'id':'other','gate':{'enabled':False,'label':'other','abstained':True}}]
        result=metrics(rows,predictions)
        self.assertEqual((result['precision'],result['recall']), (1.,1.))
        self.assertEqual(result['effective_label_matches'],1)
        self.assertTrue(result['pass'])

    def test_a_missed_activation_is_a_false_negative(self):
        rows=[{'id':'self','expected_gate':True,'expected_label':'self'}]
        predictions=[{'id':'self','gate':{'enabled':False,'label':'self','abstained':True}}]
        result=metrics(rows,predictions)
        self.assertEqual((result['tp'],result['fp'],result['fn']), (0,0,1))
        self.assertEqual(result['recall'],0.)
        self.assertIsNone(result['precision'])
        self.assertFalse(result['pass'])

    def test_wrong_membership_cannot_produce_a_complete_score(self):
        with self.assertRaises(ValueError): metrics([{'id':'a'}],[])


if __name__=='__main__':unittest.main()
