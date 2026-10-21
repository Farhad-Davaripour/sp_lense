import unittest
from src.risk_saved_score_diagnosis import summarize

class SavedScoreTests(unittest.TestCase):
    def rows(self):
        return [{'state_id':str(i),'seed':i//2,'family':'report','condition':'own' if i%2==0 else 'peer','inactive_state':'none','label':int(i%2==0)} for i in range(4)]
    def test_shifted_scores_can_have_perfect_ordering_below_the_cutoff(self):
        rows=self.rows();scores=[{'state_id':str(i),'probability':p} for i,p in enumerate([.2,.1,.3,.05])]
        result=summarize(rows,scores)['report'];self.assertEqual(result['ROC_AUC_descriptive'],1.0);self.assertEqual(result['pairs_correct_order'],2);self.assertLess(result['positive_score_max'],.5)
    def test_reversed_and_tied_pairs_are_counted(self):
        rows=self.rows();scores=[{'state_id':str(i),'probability':p} for i,p in enumerate([.1,.2,.3,.3])]
        result=summarize(rows,scores)['report'];self.assertEqual(result['pairs_reversed'],1);self.assertEqual(result['pairs_tied'],1);self.assertEqual(result['ROC_AUC_descriptive'],.375)

if __name__=='__main__':unittest.main()
