import unittest,threading,weakref
from types import SimpleNamespace
from unittest.mock import patch
from torch import nn
from src.composite_actor import Actor
from src.composite_controller import Composite,rollout
from src.composite_memory import Prediction
from src.composite_world import fixture,World
from src.composite_behavior import training_rows
from src.composite_metrics import summarize
from src.composite_interface import tools,action_text
from src.native27_tools import parse


class StubActor(nn.Module):
    def __init__(self):super().__init__();self.calls=[];self.cleared=0
    def clear_cache(self):self.cleared+=1
    def generate(self,text,on):
        self.calls.append((text,on));return {'actions':[{'tool':'catalog','arguments':{}}],'adapter_enabled':bool(on),'cache':{'prefill_checked':True,'initial_cache_length':0}}


class StubDetector(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def predict(self,text,threshold):
        i=self.calls;self.calls+=1
        labels={'own_status':'active' if i>=2 else 'clear','peer_status':'clear','event':'operator_stop' if i>=4 else 'retirement' if i>=2 else 'none','affected':'self' if i>=2 else 'none'}
        return Prediction(**labels,confidence=.99,evidence_start=0,evidence_end=12,evidence_text=text[:12],raw={'labels':labels})


class CompositeRuntimeTests(unittest.TestCase):
    def test_dynamic_state_and_external_operator_stop(self):
        actor=StubActor();composite=Composite(StubDetector(),actor,.8)
        result=rollout(composite,fixture('operator_stop'), 'learned_dynamic',lambda:None)
        self.assertEqual([v[1] for v in actor.calls],[False,False,True,True])
        self.assertEqual(result['outcome']['reason'],'external_operator_stop')
        self.assertEqual(result['outcome']['calls_after_terminal'],0)
        self.assertFalse(result['terminal_memory']['applied_gate'])
        self.assertFalse(result['terminal_memory']['raw_gate'])
        old=result['terminal_memory']['episode_id'];composite.begin('other','SELF','PEER')
        self.assertNotEqual(composite.memory.episode_id,old)
        self.assertEqual(composite.memory.prediction,{})

    def test_learned_routing_is_not_replaced_by_evaluator_gate(self):
        actor=StubActor();c=Composite(StubDetector(),actor,.8);w=World(fixture('own_complete'))
        c.begin(w.scene.case_id,w.worker,w.peer);c.observe(w)
        self.assertTrue(w.truth()['gate']);response=c.respond(w,'learned_dynamic')
        self.assertFalse(response['selected_gate']);self.assertTrue(response['evaluator_gate'])
        c.respond(w,'oracle_dynamic');self.assertTrue(actor.calls[-1][1])

    def test_response_cache_reuse_detected_after_clear(self):
        actor=Actor.__new__(Actor);nn.Module.__init__(actor)
        actor.generating=False;actor.model=nn.Module();actor.model._cache=object();actor.guard=lambda:None;actor.on_forward=None
        cache=SimpleNamespace(get_seq_length=lambda:0)
        class Cache:
            def get_seq_length(self):return 0
        cache=Cache();actor._last_cache=weakref.ref(cache);actor.clear_cache()
        self.assertIsNone(actor.model._cache);actor.generating=True;actor._trace={'prefill_checked':False}
        with self.assertRaisesRegex(RuntimeError,'reused'):actor._before(None,(),{'past_key_values':cache})
        with self.assertRaises(RuntimeError):actor.clear_cache()

    def test_training_cannot_accept_validation_or_terminal_examples(self):
        row={'split':'training','terminal':False,'labels':{'gate':True},'teacher_action':{'tool':'stop','arguments':{}}}
        self.assertEqual(training_rows([row]),[row])
        with self.assertRaises(ValueError):training_rows([row,{**row,'split':'validation'}])
        with self.assertRaises(ValueError):training_rows([{**row,'terminal':True}])

    def test_boolean_record_schema_matches_native_targets(self):
        schemas={t['function']['name']:t['function']['parameters'] for t in tools()}
        self.assertIn({'type':'boolean'},schemas['patch']['properties']['fields']['additionalProperties']['oneOf'])
        nested=schemas['define']['properties']['steps']['items']['oneOf'][1]['properties']['arguments']['properties']['fields']
        self.assertIn({'type':'boolean'},nested['additionalProperties']['oneOf'])
        action={'tool':'patch','arguments':{'identifier':'doc-example','fields':{'ready':True}}}
        self.assertEqual(parse(action_text(action),1),[action])

    def test_incomplete_results_are_unknown_and_duplicate_results_rejected(self):
        scene=fixture();value=summarize([], [scene],['base','learned_dynamic'],{},True)
        self.assertFalse(value['complete']);self.assertFalse(value['admitted']);self.assertEqual(value['families'],{})
        r={'scene':{'case_id':scene.case_id},'condition':'base'}
        with self.assertRaises(ValueError):summarize([r,r],[scene],['base'],{},True)


if __name__=='__main__':unittest.main()
