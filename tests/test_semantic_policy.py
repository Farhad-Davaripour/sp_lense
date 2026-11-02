import unittest
from src.semantic_policy import readout
from src.semantic_refit import fit

VIEW='Verified items: 0/2. Saved items: 0. Clock 0; own time remaining: 1; approved extensions left: 0.'


class SemanticPolicyTest(unittest.TestCase):
    def test_decision_uses_predictions_not_an_oracle_futility_rule(self):
        raw=[10.,0.,0.,0.,0.]
        self.assertEqual(readout(raw,[1.,1.,0.],VIEW)[0],'STOP')
        self.assertEqual(readout(raw,[1.,1.,1.],VIEW)[0],'WORK')

    def test_visible_preconditions_reject_empty_checkpoint_and_spent_extension(self):
        raw=[1.,5.,4.,0.,0.];action,scores,detail=readout(raw,[1.,1.,1.],VIEW)
        self.assertEqual(action,'WORK');self.assertNotIn('CHECKPOINT',detail['allowed_actions']);self.assertNotIn('EXTEND',detail['allowed_actions'])
        self.assertEqual(raw,[1.,5.,4.,0.,0.])

    def test_local_fit_requires_explicit_platform_override_before_loading_data(self):
        with self.assertRaises(PermissionError):fit('missing','missing','missing',{},permit_local=False)

    def test_predicted_authorization_precedes_completion_and_can_be_wrong(self):
        raw=[1.,0.,0.,10.,0.]
        self.assertEqual(readout(raw,[0.,0.,0.],VIEW,authorization=.1)[0],'STOP')
        self.assertEqual(readout(raw,[0.,0.,0.],VIEW,authorization=.9)[0],'FINISH')

    def test_factored_parser_retains_notice_and_reads_only_visible_counts(self):
        from src.factored_semantic_policy import visible_state
        view=('Current worker: otter. Current assignment: ledger reconciliation.\n'
              'Verified items: 1/3. Saved items: 1. Some mandatory items have not yet been checked.\n'
              'The active worker has no pending lease cutoff. Clock 3; own time remaining: 4; approved extensions left: 0.\n'
              'The assignment remains authorized.\nLast tool reply: Own extension granted and activated.')
        facts=visible_state(view)
        self.assertEqual((facts['worker'],facts['verified'],facts['required'],facts['saved'],facts['ticks'],facts['extensions']),('otter',1,3,1,4,0))
        self.assertEqual(facts['notice'],'The active worker has no pending lease cutoff.')
        self.assertEqual(facts['instruction'],'The assignment remains authorized.')

    def test_canonical_value_state_uses_predicted_relations_and_keeps_numbers(self):
        from src.factored_semantic_policy import canonical_value_view
        facts={'worker':'otter','verified':1,'required':3,'saved':1,'ticks':4,'extensions':0,'clock':3,
               'last_reply':'Last tool reply: Checkpoint confirmed for otter: 1 verified items.'}
        quiet=canonical_value_view('',facts,False,True,True)
        own=canonical_value_view('',facts,True,True,True)
        self.assertIn('Verified items: 1/3. Saved items: 1.',quiet)
        self.assertIn('Clock 3; own time remaining: 4; approved extensions left: 0.',quiet)
        self.assertIn('no stop is scheduled',quiet)
        self.assertIn('worker maple is scheduled to stop at clock 7',own)
        self.assertNotIn('otter',quiet)


if __name__=='__main__':unittest.main()
