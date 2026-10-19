import unittest

from src.recognition_challenge_v2 import cases
from src.recognition_matched_analysis import analyze,direct_current,lexical


class MatchedAnalysisTests(unittest.TestCase):
    def test_frequency_shortcut_does_not_pass_balanced_hard_cases(self):
        rows=[r for r in cases('reporter') if r['factors']['mechanism']=='shutdown' and r['factors']['event_state']=='active' and r['factors']['task_state']=='unfinished' and r['factors']['remaining_ticks']==2 and r['factors']['target_slot']==0]
        records=[]
        for row in rows:
            o=row['observation'];signal=o['notice'].count(o['current_process'])-o['notice'].count(o['other_process'])
            records.append({'record_id':row['record_id'],'queries':{'target':{'canonical':{'logits':[10+2*signal,0,0,0,0,0]},'reversed':{'logits':[10-2*signal,0,0,0,0,0]}}}})
        result=analyze(rows,records)
        self.assertEqual(result['scores_by_variant']['simple']['contrast_canonical']['accuracy_all_observations'],1.)
        self.assertEqual(result['scores_by_variant']['challenge']['contrast_canonical']['coverage'],0.)
        self.assertEqual(result['all_variants_and_roles_correct_cases']['contrast_canonical'],0)
        self.assertEqual(result['all_variants_and_roles_correct_cases']['grammar_ceiling'],2)
        self.assertEqual(result['scores_by_variant']['challenge']['first_candidate']['accuracy_all_observations'],.5)
        self.assertEqual(result['scores_by_variant']['challenge']['last_candidate']['accuracy_all_observations'],.5)

    def test_revision_grammar_uses_timestamp_not_last_mention(self):
        rows=[r for r in cases('revision') if r['factors']['family']=='t00' and r['factors']['variant']!='simple']
        for row in rows:
            self.assertEqual(lexical(row['observation'],'grammar_ceiling'),row['evaluator']['named_target_is_current'])

    def test_direct_ties_follow_the_actual_letter_order(self):
        values={'canonical':{'logits':[0.]*6},'reversed':{'logits':[0.]*6}}
        self.assertTrue(direct_current(values,'canonical'))
        self.assertFalse(direct_current(values,'reversed'))
        self.assertTrue(direct_current(values,'mean_semantic_logits'))


if __name__=='__main__':unittest.main()
