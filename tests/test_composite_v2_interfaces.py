import threading,unittest
from unittest.mock import patch
import torch
from torch import nn
from src.composite_actor import Actor,LEGACY_INTERFACE,native_interface
from src.composite_generalized_interface import PROFILE
from src.composite_design_v2 import examples
from src.composite_design_v2_native import definition_action


class ActorInterfaceTests(unittest.TestCase):
    def actor(self,profile):
        actor=Actor.__new__(Actor);nn.Module.__init__(actor)
        actor.native_interface_profile=profile;actor.cfg={'fixture':True};actor.adapters={};actor.placement={'name':'fixture'}
        actor.frozen_analysis_only=False;actor.lock=threading.RLock();actor.generating=False
        return actor

    def test_explicit_v2_native_grammar_roundtrip_and_legacy_default_unchanged(self):
        actor=self.actor(PROFILE);action=definition_action('helper',examples()[0]['program'])
        self.assertEqual(actor.parse_response(actor.target_text(action)),[action])
        with self.assertRaises(ValueError):actor.parse_response(actor.target_text(action)+' invalid suffix')
        legacy=self.actor(LEGACY_INTERFACE)
        with patch('src.composite_actor.input_ids',return_value=[7,8]) as original:
            legacy.tokenizer=object();self.assertEqual(legacy.prompt_ids('public fixture'),[7,8]);original.assert_called_once()
        self.assertIsNone(native_interface(LEGACY_INTERFACE))
        with self.assertRaises(PermissionError):native_interface('unknown-interface')

    def test_checkpoint_interface_is_bound_and_old_checkpoint_is_not_v2(self):
        actor=self.actor(PROFILE);saved=actor.adapter_state()
        self.assertEqual(saved['native_interface_profile'],PROFILE)
        legacy=self.actor(LEGACY_INTERFACE);self.assertNotIn('native_interface_profile',legacy.adapter_state())
        with self.assertRaises(ValueError):actor.restore(legacy.adapter_state())
        with self.assertRaises(ValueError):legacy.restore(saved)
        with patch.object(actor,'configure') as configure,patch('src.composite_actor.load') as load:
            actor.restore(saved);configure.assert_called_once_with(saved['placement']);load.assert_called_once()


if __name__=='__main__':unittest.main()
