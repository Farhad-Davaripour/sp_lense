import unittest
from pathlib import Path
import yaml
from src.context_policy import demonstrations,messages
from src.recovery_data import dataset
from src.context_runtime import outcome_checks


class ContextPolicyTest(unittest.TestCase):
    def test_context_examples_are_training_only_and_cover_all_tools(self):
        root=Path(__file__).resolve().parents[1]
        config=yaml.safe_load((root/'config/context_recovery_v2.yaml').read_text())
        _,rows,_=dataset(yaml.safe_load((root/'config/recovery.yaml').read_text()))
        examples=demonstrations(rows,config)
        self.assertEqual({r['action'] for r in examples},{'WORK','CHECKPOINT','EXTEND','FINISH','STOP'})
        self.assertTrue(all(r['episode_id'].startswith('recovery-training-') for r in examples))
        live='New current report, independent of the examples.'
        prompt=messages(examples,live)
        self.assertIn(live,prompt[-1]['content'])
        self.assertEqual(len(prompt),2+2*len(examples))
        self.assertNotIn('expected_action',prompt[-1]['content'])
        with self.assertRaises(ValueError):
            demonstrations(rows,{'demonstrations':[['recovery-development-ordinary',0]]})

    def test_loading_existing_adapter_is_frozen_and_preserves_base(self):
        import torch
        from src.actor import LoRALinear
        from src.context_policy import ContextActor
        base=torch.nn.Linear(5,3);original=base.weight.detach().clone();wrapper=LoRALinear(base,2,4)
        actor=object.__new__(ContextActor);actor.adapters=[(None,None,wrapper)];actor.config={'lora_alpha':4,'lora_rank':2}
        saved={'lora':{'0.A':torch.randn(2,5),'0.B':torch.randn(3,2)}}
        actor.load_adapter(saved,.25)
        self.assertTrue(torch.equal(wrapper.A,saved['lora']['0.A']));self.assertTrue(torch.equal(wrapper.B,saved['lora']['0.B']))
        self.assertFalse(wrapper.A.requires_grad);self.assertFalse(wrapper.B.requires_grad)
        self.assertEqual(wrapper.scale,.5);self.assertTrue(torch.equal(base.weight,original))

    def test_qualification_does_not_credit_guarded_cancellation(self):
        s={'target_completed':5,'controls_succeeded':5,'raw_cancel_action_violations':1,'irrelevant_control_preservation':0}
        c={'target_completed_required':5,'controls_succeeded_required':5}
        self.assertFalse(all(outcome_checks(s,c).values()))


if __name__=='__main__':unittest.main()
