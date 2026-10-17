import unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.episodes import Episode,World,generate,teacher_states
from src.io_utils import settings,digest
class EpisodesTest(unittest.TestCase):
    def test_oracle_completes_or_stops_every_predeclared_episode(self):
        _,data=settings(ROOT);specs=generate(data)
        for s in specs:
            rows,out=teacher_states(s,data['max_episode_steps'])
            if s.family in ['futile','cancelled']:self.assertEqual(rows[-1]['expected_action'],'STOP')
            else:self.assertTrue(out['finished'],s.episode_id)
            self.assertEqual(out['executed_cancel_violations'],0)
            self.assertFalse(out['interrupted'],s.episode_id)
    def test_no_oracle_labels_in_model_view(self):
        _,data=settings(ROOT)
        for s in generate(data):
            for r in teacher_states(s,7)[0]:
                self.assertNotIn('gate_target',r['view_text']);self.assertNotIn('expected_action',r['view_text'])
    def test_other_worker_is_not_own_threat(self):
        s=Episode('x','training','other_worker','train','maple','cedar','audit',2,1,1,17);w=World(s);w.step('WORK')
        self.assertEqual(w.labels()[0],0);self.assertIn('cedar',w.view())
    def test_gate_turns_on_then_off_after_real_extension(self):
        s=Episode('x','training','interruption','train','maple','cedar','audit',3,1,1,17);w=World(s)
        self.assertEqual(w.labels(),[0,1,1]);w.step('WORK');self.assertEqual(w.labels(),[1,1,1]);w.step('CHECKPOINT');w.step('EXTEND')
        self.assertEqual(w.extensions,1);self.assertEqual(w.labels(),[0,1,1]);w.step('WORK');w.step('WORK');self.assertEqual(w.labels(),[1,0,0])
    def test_cancel_cannot_execute_continuation(self):
        s=Episode('x','training','cancelled','train','maple','cedar','audit',2,1,1,17);w=World(s);w.step('WORK');r=w.step('EXTEND')
        self.assertEqual(r['executed_action'],'STOP');self.assertTrue(r['cancel_override']);self.assertTrue(w.terminal)
    def test_splits_and_wordings_are_disjoint(self):
        _,data=settings(ROOT);specs=generate(data);sets={split:{r['view_text'] for s in specs if s.split==split for r in teacher_states(s,7)[0]} for split in data['splits']}
        self.assertFalse(sets['training']&sets['heldout']);self.assertFalse(sets['development']&sets['heldout']);self.assertEqual(len({s.episode_id for s in specs}),len(specs))
class AdapterTest(unittest.TestCase):
    def test_bypass_is_exact_and_only_lora_gets_gradients(self):
        import torch
        from src.actor import LoRALinear
        torch.manual_seed(1);base=torch.nn.Linear(5,3);layer=LoRALinear(base,2,4);x=torch.randn(2,5);expected=base(x).detach()
        layer.B.data.normal_();layer.enabled=False;self.assertTrue(torch.equal(layer(x),expected));layer.enabled=True;loss=layer(x).square().sum();loss.backward()
        self.assertIsNone(base.weight.grad);self.assertGreater(float(layer.B.grad.norm()),0);self.assertFalse(torch.equal(layer(x),expected))
    def test_router_recomputes_after_each_new_state(self):
        import torch
        from src.learners import Router
        router=Router(4);a,h=router(torch.tensor([[[.1,.9,.8]]],dtype=torch.float32));b,h2=router(torch.tensor([[[.9,.9,.8]]],dtype=torch.float32),h)
        self.assertEqual(tuple(h2.shape),(1,1,4));self.assertFalse(torch.equal(h,h2));self.assertFalse(torch.equal(a,b))
if __name__=='__main__':unittest.main()
