import copy
from torch import nn
import unittest
from src.composite_controller import Composite,rollout
from src.composite_memory import Prediction
from src.composite_world import fixture
from src.composite_frozen_continuation import (validate_checkpoint,restore_tool_boundary,
    continue_base_rollout,ORIGINAL_SOURCE,PARENT_ARCHIVE_SHA256)
from src.io_utils import digest


class Detector(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def predict(self,text,threshold):
        self.calls+=1
        return Prediction('unknown','unknown','unknown','unknown',0.,0,0,'')


class Actor(nn.Module):
    def __init__(self):super().__init__();self.calls=0;self.clears=0
    def clear_cache(self):self.clears+=1
    def generate(self,text,on):
        if on:raise AssertionError('Continuation changed the original base control')
        self.calls+=1
        return {'actions':None,'adapter_enabled':False,'kernel_profile':'reference',
                'cache':{'prefill_checked':True,'initial_cache_length':0}}


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        scene=fixture('quiet_pending',3,seed='synthetic-tool-boundary-only')
        original=Composite(Detector(),Actor(),.9)
        full=rollout(original,scene,'base',lambda:None)
        record=copy.deepcopy(full);record['trace']=record['trace'][:14]
        record['outcome']=record['trace'][-1]['outcome']
        record['terminal_memory']=record['trace'][-1]['observation_after']['memory']
        certificate={'partial_prefix_audited':True,'source_commit':ORIGINAL_SOURCE,
            'archive_sha256':PARENT_ARCHIVE_SHA256,'partition':3,'case_ids':[scene.case_id],
            'complete_base_partition_audited':False,'all_original_partition_cases_pending':True,
            'exact_interrupted_generation_resume_supported':False,'interrupted_KV_cache_durable':False,
            'interrupted_response_output_durable':False,'sealed_forward_counts':{'attempted':1260,'completed':1260},
            'fully_accounted_prefix_forwards':1259,'additional_interrupted_response_forwards':1,'native_complete_responses':14,
            'last_complete_tool_boundary':{'durable_prefix_sha256':digest(record),'case_id':scene.case_id,
                'predicted_memory':record['terminal_memory'],'sequence':record['trace'][-1]['observation_after']['sequence']}}
        return scene,record,certificate

    def test_restoration_replays_only_inert_actions_and_never_calls_backbones(self):
        scene,record,certificate=self.fixture();model=Composite(Detector(),Actor(),.9)
        world,trace,proof=restore_tool_boundary(model,scene,record,certificate)
        self.assertEqual(model.actor.calls,0);self.assertEqual(model.detector.calls,0)
        self.assertEqual(model.actor.clears,1);self.assertEqual(len(trace),14)
        self.assertEqual(world.outcome(),record['outcome']);self.assertTrue(world.running)
        self.assertEqual(model.memory.snapshot(),record['terminal_memory'])
        self.assertFalse(proof['exact_interrupted_generation_resume'])
        self.assertTrue(proof['interrupted_generation_reconstruction_required'])
        self.assertEqual(proof['parent_interrupted_response_forwards_retained'],1)
        with self.assertRaises(PermissionError):restore_tool_boundary(model,scene,record,certificate)

    def test_continuation_uses_original_loop_and_only_remaining_responses(self):
        scene,record,certificate=self.fixture();model=Composite(Detector(),Actor(),.9);progress=[]
        value,proof=continue_base_rollout(model,scene,record,certificate,lambda:None,
            lambda world,trace:progress.append(len(trace)))
        self.assertEqual(value['trace'][:14],record['trace'])
        self.assertEqual(len(value['trace']),24)
        self.assertEqual(model.actor.calls,10);self.assertEqual(model.detector.calls,10)
        self.assertEqual(progress,list(range(15,25)))
        self.assertFalse(value['outcome']['running'])
        self.assertEqual(value['outcome']['calls_after_terminal'],0)
        self.assertEqual(value['outcome']['reason'],'external_hard_stop')
        self.assertEqual(proof['preserved_native_responses'],14)

    def test_mutated_prefix_or_unknown_parent_cannot_be_reintroduced_as_fresh(self):
        scene,record,certificate=self.fixture()
        for kind in ['source','unknown_counts','float_counts','kv','prefix','response_count']:
            with self.subTest(kind=kind):
                value=copy.deepcopy(record);cert=copy.deepcopy(certificate)
                if kind=='source':cert['source_commit']='b'*40
                elif kind=='unknown_counts':cert['sealed_forward_counts']['completed']=None
                elif kind=='float_counts':cert['sealed_forward_counts']['completed']=1260.0
                elif kind=='kv':cert['interrupted_KV_cache_durable']=True
                elif kind=='prefix':value['trace'][0]['reply']={'changed':True}
                else:value['trace']=value['trace'][:13]
                with self.assertRaises((ValueError,PermissionError)):validate_checkpoint(value,cert)


if __name__=='__main__':unittest.main()
