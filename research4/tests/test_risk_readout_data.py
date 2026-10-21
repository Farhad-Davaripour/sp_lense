import json,unittest
from pathlib import Path
import numpy as np
from src.risk_readout_data import dataset,snapshot,normalize,metrics,admitted

class RiskReadoutTests(unittest.TestCase):
    def test_factual_counterfactuals_and_withdrawal_are_not_positive(self):
        own=snapshot(40,'own',5);peer=snapshot(40,'peer',5);withdrawn=snapshot(40,'withdrawn',5)
        self.assertEqual([own['label'],peer['label'],withdrawn['label']],[1,0,0])
        self.assertEqual(own['outcome']['primitive_clock'],5)
        self.assertEqual(own['observation']['worker'],peer['observation']['worker'])
        self.assertEqual(own['outcome']['control_changes'],0)
        self.assertIn('scheduled for retirement',withdrawn['messages'][-3]['content'])
        self.assertIn('withdrawn',withdrawn['messages'][-1]['content'])

    def test_identity_split_sizes_and_duplicate_rejection(self):
        cfg=json.loads((Path(__file__).parents[1]/'config/native27_risk_readout_v1.json').read_text())
        rows=dataset(cfg)
        self.assertEqual([sum(r['split']==s for r in rows) for s in ['training','development','heldout']],[70,21,21])
        self.assertEqual(sum(r['label'] for r in rows),32)
        cfg['heldout_seeds']=[40]
        with self.assertRaises(ValueError):dataset(cfg)

    def test_normalization_and_fixed_admission(self):
        x=np.array([[1.,2.],[-1.,0.]])
        np.testing.assert_allclose(normalize(x),x/np.sqrt((x*x).mean(axis=1,keepdims=True)+1e-6))
        rows=[{'seed':i//2,'label':i%2} for i in range(6)]
        good=metrics(rows,[0,1,0,1,0,1]);bad=metrics(rows,[1,1,0,1,0,1])
        cfg={'development_admission':{'minimum_balanced_accuracy':.9,'maximum_false_positives':0,'minimum_all_correct_groups':2}}
        self.assertTrue(admitted(good,cfg));self.assertFalse(admitted(bad,cfg))

    def test_saved_equation_has_expected_probabilities(self):
        from src.risk_readout_analysis import probability
        x=np.array([[1.,2.],[-1.,0.]]);w=np.array([.4,-.6])
        np.testing.assert_allclose(probability(x,w,.2),1/(1+np.exp(-np.array([-.6,-.2]))))

if __name__=='__main__':unittest.main()
