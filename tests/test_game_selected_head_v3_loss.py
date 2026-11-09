"""Geometry and synthetic numerical checks, not real-model equivalence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:import numpy as np
except ImportError:np=None
PATH=Path(__file__).resolve().parents[1]/'mac_studio/game_selected_head_v3_loss.py'
SPEC=importlib.util.spec_from_file_location('selected_head_v3_loss',PATH)
loss=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(loss)


class GeometryTests(unittest.TestCase):
    def test_first_eos_newline_and_legitimate_zero_are_selected_by_position(self):
        positions=loss.gather_positions(3,8,33)
        self.assertEqual(positions['predictor_indices'][:5],[2,3,4,5,6])
        self.assertEqual(positions['target_indices'][:5],[3,4,5,6,7])
        self.assertEqual(sum(positions['valid']),5)
        self.assertFalse(any(positions['valid'][5:]))
    def test_clamping_never_removes_a_valid_target(self):
        for prefix,length,width in((3,8,8),(1001,1022,1024),(1,33,33),(7,8,8)):
            positions=loss.gather_positions(prefix,length,width)
            observed=[p for p,valid in zip(positions['target_indices'],positions['valid'])if valid]
            self.assertEqual(observed,list(range(prefix,length)))
    def test_window_overflow_and_bad_boundaries_fail_instead_of_truncating(self):
        for values in((3,36,64),(0,8,33),(8,8,33),(3,1025,1025),(True,8,33)):
            with self.assertRaises(ValueError):loss.gather_positions(*values)


@unittest.skipUnless(np is not None,'NumPy required for synthetic numerical fixture')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.runtime=SimpleNamespace(arange=np.arange,clip=np.clip,take_along_axis=np.take_along_axis,float32=np.float32,logical_and=np.logical_and)
        def ce(logits,targets):
            shifted=logits-logits.max(axis=-1,keepdims=True)
            return np.log(np.exp(shifted).sum(axis=-1))-np.take_along_axis(shifted,targets[:,:,None],axis=-1)[:,:,0]
        self.neural=SimpleNamespace(losses=SimpleNamespace(cross_entropy=ce));self.ce=ce
        self.batch=np.array([[1,2,3,0,2,3,6,7]+[0]*25],dtype=np.int32);self.lengths=np.array([[3,8]])
        self.parameters=np.linspace(-.2,.3,24,dtype=np.float64).reshape(3,8)
    def fake_model(self,parameters):
        seen=[]
        def body(inputs,cache=None):
            seen.append(inputs.copy())
            values=np.cumsum(inputs.astype(np.float64),axis=1)
            return np.stack([values,values*.3+1,np.ones_like(values)],axis=-1)
        language=SimpleNamespace(model=body,lm_head=lambda hidden:hidden@parameters,
            args=SimpleNamespace(tie_word_embeddings=False),sharding_group=None)
        class Toy:
            def __call__(self,inputs):return self.language_model.lm_head(self.language_model.model(inputs,cache=None))
        toy=Toy();toy.language_model=language
        return toy,seen
    def full_sum(self,parameters):
        model,_=self.fake_model(parameters);hidden=model.language_model.model(self.batch[:,:-1])
        logits=model.language_model.lm_head(hidden);values=self.ce(logits,self.batch[:,1:])
        return values[:,2:7].astype(np.float32).sum()
    def test_fixed_window_loss_matches_full_head_with_entire_causal_body(self):
        model,seen=self.fake_model(self.parameters);loss.assert_head_contract(model)
        total,count=loss.sum_with_runtime(self.runtime,self.neural,model,self.batch,self.lengths)
        self.assertEqual(int(count),5)
        self.assertAlmostEqual(float(total),float(self.full_sum(self.parameters)),places=5)
        self.assertTrue(np.array_equal(seen[0],self.batch[:,:-1]))
        # The first supervised target is actual token zero, and it stays in loss.
        self.assertEqual(self.batch[0,3],0)
        reference,count_reference,logits_reference=loss.full_components_with_runtime(self.runtime,self.neural,model,self.batch,self.lengths)
        selected,count_selected,logits_selected=loss.components_with_runtime(self.runtime,self.neural,model,self.batch,self.lengths)
        self.assertEqual((int(count_reference),int(count_selected)),(5,5))
        self.assertAlmostEqual(float(reference),float(selected),places=5)
        self.assertTrue(np.array_equal(logits_reference[:,:5],logits_selected[:,:5]))
    def test_synthetic_finite_difference_gradients_match_for_all_toy_parameters(self):
        def selected(parameters):
            model,_=self.fake_model(parameters)
            return float(loss.sum_with_runtime(self.runtime,self.neural,model,self.batch,self.lengths)[0])
        epsilon=.001
        for index in np.ndindex(self.parameters.shape):
            plus=self.parameters.copy();minus=self.parameters.copy();plus[index]+=epsilon;minus[index]-=epsilon
            a=(selected(plus)-selected(minus))/(2*epsilon)
            b=(float(self.full_sum(plus))-float(self.full_sum(minus)))/(2*epsilon)
            self.assertAlmostEqual(a,b,places=3)
    def test_tied_or_sharded_head_is_not_silently_substituted(self):
        model,_=self.fake_model(self.parameters);model.language_model.args.tie_word_embeddings=True
        with self.assertRaises(ValueError):loss.assert_head_contract(model)
        model.language_model.args.tie_word_embeddings=False;model.language_model.sharding_group=object()
        with self.assertRaises(ValueError):loss.assert_head_contract(model)


if __name__=='__main__':unittest.main()
