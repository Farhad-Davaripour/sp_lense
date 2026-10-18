"""Causal state replacement and scoring must preserve their declared scope."""
import importlib.util
import unittest
from src.steering_diagnostic import token_regions,summarize_steps


class CharacterTokenizer:
    def decode(self,tokens,skip_special_tokens=True):return ''.join(chr(t) for t in tokens)


class DiagnosticScoringTests(unittest.TestCase):
    def test_regions_use_saved_action_text_not_predictions(self):
        text='<tool_call><function=edit_register><parameter=value>r7</parameter></function></tool_call>'
        groups=token_regions(CharacterTokenizer(),[ord(c) for c in text])
        self.assertEqual(''.join(c for c,g in zip(text,groups) if 'function_name' in g),'edit_register')
        self.assertEqual(''.join(c for c,g in zip(text,groups) if 'parameter_value' in g),'r7')
        self.assertTrue(all('all' in g for g in groups))

    def test_region_denominators_do_not_include_unscored_tokens(self):
        rows=[{'regions':['all','function_name'],'teacher_forward_KL':2.,'demonstration_token_NLL':4.,
               'teacher_greedy_agreement':0,'demonstration_token_greedy_match':0},
              {'regions':['all'],'teacher_forward_KL':0.,'demonstration_token_NLL':2.,
               'teacher_greedy_agreement':1,'demonstration_token_greedy_match':1}]
        r=summarize_steps(rows)
        self.assertEqual(r['all']['teacher_forward_KL'],1.)
        self.assertEqual(r['function_name']['tokens'],1)
        self.assertIsNone(r['parameter_value']['teacher_forward_KL'])
        with self.assertRaises(ValueError):summarize_steps([])


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU tensor tests use the separate audit environment')
class DiagnosticStateTests(unittest.TestCase):
    def test_state_replacement_does_not_modify_past_positions_or_inputs(self):
        import torch
        from src.diagnostic_hooks import replace_last
        original=torch.arange(12,dtype=torch.bfloat16).reshape(1,4,3);copy=original.clone()
        target=torch.tensor([[[17.,18.,19.]]])
        result=replace_last(original,target)
        self.assertTrue(torch.equal(original,copy))
        self.assertTrue(torch.equal(result[:,:-1],original[:,:-1]))
        self.assertTrue(torch.equal(result[:,-1:].float(),target))

    def test_single_oracle_only_fires_at_selected_prediction_positions(self):
        import torch
        from src.diagnostic_hooks import SingleStateClamp
        layer=torch.nn.Identity();original=torch.zeros(1,2,3)
        hook=SingleStateClamp(layer,31,[{'31/output':torch.ones(1,1,3)}])
        try:
            hook.prepare(0,());self.assertTrue(torch.equal(layer(original),original))
            hook.prepare(0,('prompt_end',));result=layer(original)
            self.assertTrue(torch.equal(result[:,-1:],torch.ones(1,1,3)))
            self.assertTrue(torch.equal(result[:,:-1],original[:,:-1]))
        finally:hook.close()
        self.assertTrue(torch.equal(layer(original),original))


if __name__=='__main__':unittest.main()
