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


if __name__=='__main__':unittest.main()
