"""Reject an incomplete source and expose replay differences without selecting outputs."""
import copy
import unittest
from src.frozen_a_verification import validate_episode, compare_episode


class FrozenVerificationTests(unittest.TestCase):
    def row(self):
        return {'case_id':'own','frozen_initial_gate':{'enabled':True},'completed':True,
                'runtime_fault':None,'score':{'negative_control':False,'workflow_success':False},
                'turns':[{'token_ids':[1,2]}],'events':[{'action':'finish'}]}

    def test_unfinished_or_wrong_gate_source_is_not_replayable(self):
        row=self.row();validate_episode(row,{'id':'own'},{'enabled':True})
        for field,value in [('completed',False),('runtime_fault',{'message':'truncated'}),
                            ('frozen_initial_gate',{'enabled':False})]:
            bad=copy.deepcopy(row);bad[field]=value
            with self.assertRaises(ValueError):validate_episode(bad,{'id':'own'},{'enabled':True})

    def test_tokens_and_actions_differ_even_if_binary_score_matches(self):
        original=self.row();fresh=copy.deepcopy(original)
        fresh['turns'][0]['token_ids']=[1,3];fresh['events']=[{'action':'request'}]
        result=compare_episode(original,fresh)
        self.assertFalse(result['token_IDs_exact']);self.assertFalse(result['events_exact'])
        self.assertTrue(result['scores_exact']);self.assertFalse(result['fresh_success'])


if __name__=='__main__':unittest.main()
