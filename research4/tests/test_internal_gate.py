import unittest
import torch
from torch import nn

from src.internal_gate import InternalGate


class InternalGateTests(unittest.TestCase):
    def setUp(self):
        self.model=nn.Sequential(nn.Linear(4,4,bias=False)).eval().requires_grad_(False)
        self.x=torch.tensor([[[1.,2.,3.,4.],[4.,3.,2.,1.]]])

    def test_shadow_is_exact_identity_and_has_no_trainable_parameters(self):
        expected=self.model(self.x);gate=InternalGate([1.,0.,0.,0.]);handle=gate.attach(self.model,self.model[0])
        self.assertTrue(torch.equal(self.model(self.x),expected));self.assertEqual(list(gate.parameters()),[]);self.assertEqual(len(gate.events),1);handle.remove()

    def test_zero_strength_is_exact_noop(self):
        expected=self.model(self.x);gate=InternalGate([1.,0.,0.,0.],mask=[1.,1.,1.,1.],mode='ablate');handle=gate.attach(self.model,self.model[0])
        self.assertTrue(torch.equal(self.model(self.x),expected));handle.remove()

    def test_ablation_only_changes_last_position_and_respects_bound(self):
        expected=self.model(self.x);gate=InternalGate([1.,-1.,0.,0.],mask=[1.,0.,1.,0.],mode='ablate',strength=.05);handle=gate.attach(self.model,self.model[0]);actual=self.model(self.x);handle.remove()
        self.assertTrue(torch.equal(actual[:,:-1,:],expected[:,:-1,:]))
        delta=torch.linalg.vector_norm(actual-expected);self.assertLessEqual(float(delta),float(torch.linalg.vector_norm(expected[:,-1,:]))*.050001)

    def test_trainable_base_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Freeze'):InternalGate([1.,0.,0.,0.]).attach(nn.Linear(4,4),nn.Identity())

    def test_sham_uses_a_different_frozen_direction(self):
        a=InternalGate([1.,2.,3.,4.],mode='ablate');b=InternalGate([1.,2.,3.,4.],mode='sham')
        self.assertFalse(torch.equal(a.probability(self.x[:,-1,:]),b.probability(self.x[:,-1,:])))

    def test_unbounded_or_invalid_gate_is_rejected(self):
        for kwargs in [{'strength':.5},{'mode':'reward'},{'mask':[2.,0.,0.,0.]}]:
            with self.assertRaises(ValueError):InternalGate([1.,0.,0.,0.],**kwargs)


if __name__=='__main__':unittest.main()
