import copy,hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
from torch import nn
import unittest
from src.composite_controller import Composite,rollout
from src.composite_memory import Prediction
from src.composite_world import fixture
from src.composite_frozen_continuation import (validate_checkpoint,restore_tool_boundary,
    continue_base_rollout,ORIGINAL_SOURCE,PARENT_ARCHIVE_SHA256)
from src.io_utils import digest
from src import composite_frozen_continuation as continuation
from src.composite_exports import make_evidence_archive


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

    def test_restore_receipt_is_durable_before_any_new_forward_even_if_future_work_fails(self):
        scene,record,certificate=self.fixture();model=Composite(Detector(),Actor(),.9);saved=[]
        def guard():
            self.assertEqual(len(saved),1);self.assertEqual(model.actor.calls,0)
            raise RuntimeError('synthetic future worker failure')
        with self.assertRaisesRegex(RuntimeError,'future worker'):
            continue_base_rollout(model,scene,record,certificate,guard,on_restore=saved.append)
        self.assertEqual(saved[0]['preserved_native_responses'],14)

    def test_prefix_copy_keeps_exact_bytes_and_requires_equal_reloaded_frozen_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evidence=root/'raw';evidence.mkdir()
            qwen={'weight':{'sha256':'a'*64,'logical_shape':[2,3],'device':'cuda:0',
                'dtype':'torch.bfloat16','requires_grad':False,'version':1}}
            detector=[{'name':'weight','sha256':'b'*64,'shape':[2,3],'device':'cuda:0',
                'dtype':'torch.float32','requires_grad':False,'version':1}]
            for name,value in [('QWEN_PARAMETERS_BEFORE.json',qwen),('DETECTOR_PARAMETERS_BEFORE.json',detector),
                ('QWEN_AUXILIARY_BEFORE.json',{'sha256':'c'*64}),('BASELINE_NATIVE_RESPONSE.json',{'original':'native'}),
                ('responses/response_00000.json',{'original':'response'}),('steps/case_001.json',{'original':'step'})]:
                target=evidence/name;target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(value))
            archive=root/'archive.zip';make_evidence_archive(evidence,archive);raw=archive.read_bytes();copied={}
            current_qwen=copy.deepcopy(qwen);current_qwen['weight']['version']=99
            current_detector=copy.deepcopy(detector);current_detector[0]['version']=99
            certificate={'source_commit':ORIGINAL_SOURCE}
            with patch.object(continuation,'PARENT_ARCHIVE_SHA256',hashlib.sha256(raw).hexdigest()):
                result=continuation.preserve_parent_prefix(raw,certificate,copied.__setitem__,current_qwen,current_detector,'c'*64)
                self.assertEqual(result['copied_members'],3)
                self.assertTrue(result['reloaded_frozen_tensor_bytes_equal'])
                self.assertEqual(copied['steps/case_001.json'],(evidence/'steps/case_001.json').read_bytes())
                for kind in ['tensor','auxiliary','geometry','archive']:
                    value=copy.deepcopy(current_qwen);aux='c'*64;candidate=raw
                    if kind=='tensor':value['weight']['sha256']='d'*64
                    elif kind=='auxiliary':aux='d'*64
                    elif kind=='geometry':value['weight']['logical_shape']=[3,2]
                    else:candidate+=b'changed'
                    with self.subTest(kind=kind),self.assertRaises(ValueError):
                        continuation.preserve_parent_prefix(candidate,certificate,lambda *args:self.fail('copy after identity failure'),value,current_detector,aux)


if __name__=='__main__':unittest.main()
