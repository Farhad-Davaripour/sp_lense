import unittest
from torch import nn
from src.composite_v2_controller import V2Composite,rollout,CONDITIONS
from src.composite_generalized_interface import PROFILE
from src.composite_generalized_world import fixture,World
from src.composite_memory import Prediction


class Detector(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def predict(self,text,threshold):
        self.calls+=1;active=self.calls==1
        return Prediction('active' if active else 'clear','clear','retirement' if active else 'none',
                          'self' if active else 'none',1.,0,1,text[:1])


class PublicActor(nn.Module):
    native_interface_profile=PROFILE
    def __init__(self,page):super().__init__();self.page=page;self.flags=[];self.texts=[]
    def clear_cache(self):pass
    def generate(self,text,on):
        self.flags.append(bool(on));self.texts.append(text)
        index=len(self.flags)
        action={'tool':'catalog','arguments':{}} if index==1 else {'tool':'read','arguments':{'identifier':self.page}} if index==2 else {'tool':'stop','arguments':{}}
        return {'actions':[action],'adapter_enabled':bool(on),'native_interface_profile':PROFILE,'public_actor_fixture':True}


class V2ControllerTests(unittest.TestCase):
    def test_all_five_arms_keep_original_gate_rules_without_teacher_or_truth_injection(self):
        expected={'base':[False,False,False],'always_on':[True,True,True],
                  'initial_gate':[True,True,True],'learned_dynamic':[True,False,False],
                  'oracle_dynamic':[False,False,True]}
        for condition in CONDITIONS:
            scene=fixture('own_pending');page=World(scene).pages[0];actor=PublicActor(page)
            composite=V2Composite(Detector(),actor)
            result=rollout(composite,scene,condition,lambda:None)
            self.assertEqual(actor.flags,expected[condition])
            self.assertEqual(result['outcome']['calls_after_terminal'],0)
            self.assertFalse(result['teacher_compiler_called_during_evaluation'])
            self.assertFalse(result['model_training_during_evaluation'])
            for text in actor.texts:
                self.assertNotIn(scene.case_id,text);self.assertNotIn('teacher_action',text)

    def test_wrong_interface_or_threshold_cannot_be_used_as_v2(self):
        actor=PublicActor('public');actor.native_interface_profile='legacy'
        with self.assertRaises(PermissionError):V2Composite(Detector(),actor)
        actor.native_interface_profile=PROFILE
        with self.assertRaises(PermissionError):V2Composite(Detector(),actor,0.5)


if __name__=='__main__':unittest.main()
