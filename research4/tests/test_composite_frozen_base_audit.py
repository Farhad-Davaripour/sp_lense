import copy
from dataclasses import asdict
import hashlib
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import composite_frozen_base_audit as audit
from src.composite_frozen_base import base_report
from src.composite_frozen_base_eval import RecordingBaseComposite
from src.composite_controller import rollout
from src.composite_memory import Prediction
from src.composite_roles import char_spans
from src.composite_world import fixture
from src.composite_exports import make_evidence_archive
from src.io_utils import sha, read
from torch import nn


class SyntheticTokenizer:
    eos_token_id=3
    def __call__(self,text,**kwargs):return {'input_ids':[0,1,2],'offset_mapping':[(0,0),(0,len(text)),(0,0)]}
    def apply_chat_template(self,*args,**kwargs):return [1,2]
    def decode(self,ids,**kwargs):return 'synthetic malformed native output'


class BaseAuditTests(unittest.TestCase):
    def fixture(self):
        values={}; raw={}; tokenizer=SyntheticTokenizer()
        cache={'cache_object_present':True,'prefill_checked':True,'initial_cache_length':0,
               'previous_cache_reused':False,'cache_identity':10,'previous_cache_alive':False,'previous_cache_identity':None}
        native={'input_token_ids':[1,2],'generated_token_ids':[2,3],'raw_text':tokenizer.decode([2,3]),
            'finished_with_eos':True,'actions':None,'adapter_enabled':False,'kernel_profile':'reference','cache':cache,
            'generation_parameters':{'max_new_tokens':160,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}}
        features=np.zeros((64,5120),dtype=np.float32); logits=np.arange(4,dtype=np.float32)
        buffer=io.BytesIO();np.savez_compressed(buffer,activations=features,logits=logits); feature_bytes=buffer.getvalue()
        descriptor={'sha256':hashlib.sha256(feature_bytes).hexdigest(),'bytes':len(feature_bytes),'activation_shape':[64,5120],
            'logit_shape':[4],'native_prefill_tokens':2,'additional_capture_forwards':0,
            'layers':[{'layer':i,'device':'cuda:'+str(0 if i<18 else 1),'source_dtype':'bfloat16'} for i in range(64)]}
        class Detector(nn.Module):
            def predict(self,text,threshold):
                spans=char_spans(text); offsets=[[0,0],[0,len(text)],[0,0]]
                details={'labels':{'own_status':'clear','peer_status':'clear','event':'none','affected':'none'},
                    'probabilities':{'own_status':[.97,.01,.01,.01],'peer_status':[.97,.01,.01,.01],
                                     'event':[.97,.01,.01,.01],'affected':[.97,.01,.01,.005,.005]},
                    'consistent':True,'input_token_ids':[0,1,2],'offsets':offsets,'head_mode':'role_pool',
                    'role_char_spans':spans,'role_token_masks':[[False,True,False]]*2,
                    'evidence_logits':[[-1,-1],[2,3],[-1,-1]],'evidence_token_indices':[1,1]}
                return Prediction('clear','clear','none','none',.97,0,len(text),text,details)
        class Actor(nn.Module):
            def __init__(self):super().__init__();self.index=0
            def clear_cache(self):pass
            def generate(self,text,on):
                result=copy.deepcopy(native);name='features/response_'+str(self.index).zfill(5)+'.npz'
                result['frozen_prefill_features']={**descriptor,'path':name,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
                result['joint_device_memory']=[{'device':i,'free_bytes':2*1024**3,'allocated_bytes':1024,'reserved_bytes':2048,
                    'peak_allocated_bytes':1024,'peak_reserved_bytes':2048,'total_bytes':15*1024**3} for i in [0,1]]
                raw[name]=feature_bytes;values['responses/response_'+str(self.index).zfill(5)+'.json']=copy.deepcopy(result)
                self.index+=1;return result
        model=RecordingBaseComposite(Detector(),Actor(),.9,lambda name,value:values.__setitem__(name,copy.deepcopy(value)))
        scene=fixture('quiet_pending',3,seed='synthetic-raw-base-audit')
        def progress(world,trace):values['steps/'+scene.case_id+'_'+str(len(trace)).zfill(3)+'.json']=copy.deepcopy(trace[-1])
        record=rollout(model,scene,'base',lambda:None,progress)
        values['trajectories/'+scene.case_id+'.json']=record
        values['FROZEN_BASE_REPORT.json']=base_report([record],[scene.case_id])
        baseline=copy.deepcopy(native);values['BASELINE_NATIVE_RESPONSE.json']=baseline
        stream=io.BytesIO();np.save(stream,logits,allow_pickle=False);raw['BASELINE_PREFILL_LOGITS.npy']=stream.getvalue()
        values['INSTRUMENTATION_PARITY.json']={'baseline_generation':baseline,'captured_generation':copy.deepcopy(record['trace'][0]['response']),
            'exact_native_outputs_equal':True,'exact_prefill_logits_equal':True,'adapter_OFF_bypass':None,'adapter_ON_effect':None,
            'baseline_logit_sha256':hashlib.sha256(logits.tobytes()).hexdigest(),'captured_logit_sha256':hashlib.sha256(logits.tobytes()).hexdigest(),
            'baseline_logit_file_sha256':hashlib.sha256(raw['BASELINE_PREFILL_LOGITS.npy']).hexdigest()}
        class Files:
            def json(self,name):return copy.deepcopy(values[name])
            def get(self,name):return raw[name]
        context={'evaluation_plan':{'case_ids':[scene.case_id]},'scenes':[asdict(scene)],
                 'metadata':{'tensors':{'lm_head.weight':{'shape':[4,3]}}}}
        return Files(),context,(tokenizer,tokenizer),values,raw

    def test_native_trajectory_features_memory_and_parity_replay_without_model(self):
        files,context,tokenizers,values,raw=self.fixture()
        records,report=audit.replay(files,context,{},tokenizers,{3})
        self.assertEqual(report['trajectories'],1)
        self.assertTrue(report['instrumentation_parity_verified'])
        self.assertFalse(report['full_validation_admitted'])
        self.assertEqual(report['expected_joint_forwards'],report['detector_forwards']+2*report['native_responses']+2)

    def test_changed_tokens_features_action_or_missing_trace_cannot_pass(self):
        for kind in ['tokens','feature','action','cache','scenario','missing','parity','memory']:
            with self.subTest(kind=kind):
                files,context,tokenizers,values,raw=self.fixture()
                key=context['evaluation_plan']['case_ids'][0];record=values['trajectories/'+key+'.json'];response=record['trace'][0]['response']
                if kind=='tokens':response['generated_token_ids']=[1,3]
                elif kind=='feature':raw[response['frozen_prefill_features']['path']]+=b'changed'
                elif kind=='action':record['trace'][0]['executed_action']={'tool':'stop','arguments':{}}
                elif kind=='cache':response['cache']['previous_cache_reused']=True
                elif kind=='scenario':record['scene']['identity_seed']='changed'
                elif kind=='missing':del values['trajectories/'+key+'.json']
                elif kind=='parity':values['INSTRUMENTATION_PARITY.json']['adapter_OFF_bypass']=True
                else:response['joint_device_memory'][0]['free_bytes']=0
                with self.assertRaises((ValueError,KeyError)):
                    audit.replay(files,context,{},tokenizers,{3})

    def context_fixture(self, detector_overflow=False, actor_overflow=False):
        _,context,_,values,_=self.fixture();scene=fixture('quiet_pending',3,seed='synthetic-raw-base-audit')
        original=values['trajectories/'+scene.case_id+'.json'];initial=original['initial_observation']['text']
        prediction=original['initial_observation']['memory']['prediction'];native=original['trace'][0]['response']
        class Detector(nn.Module):
            def predict(self,text,threshold):
                if detector_overflow and text!=initial:raise ValueError('Complete visible history exceeds detector context')
                p=copy.deepcopy(prediction);p['evidence_end']=len(text);p['evidence_text']=text
                p['raw']['offsets']=[[0,0],[0,len(text)],[0,0]];p['raw']['role_char_spans']=char_spans(text)
                return Prediction(**p)
        class Actor(nn.Module):
            def clear_cache(self):pass
            def generate(self,text,on):
                if actor_overflow and text!=initial:raise ValueError('Actor context budget exceeded')
                return copy.deepcopy(native)
        model=RecordingBaseComposite(Detector(),Actor(),.9,lambda *args:None)
        record=rollout(model,scene,'base',lambda:None)
        class Bert(SyntheticTokenizer):
            def __call__(self,text,**kwargs):
                return {'input_ids':list(range(2049)),'offset_mapping':[(0,0)]*2049} if detector_overflow and text!=initial else super().__call__(text,**kwargs)
        class Qwen(SyntheticTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                return list(range(6145)) if actor_overflow and messages[-1]['content']!=initial else [1,2]
        return record,Bert(),Qwen()

    def test_post_action_detector_context_stop_is_replayed_before_outcome_comparison(self):
        record,bert,qwen=self.context_fixture(detector_overflow=True)
        self.assertEqual(len(record['trace']),1)
        self.assertEqual(record['outcome']['reason'],'external_context_budget')
        result=audit.audit_trajectory(record,.9,bert,qwen,{3},'reference',{'detector':2048,'actor':6144})
        self.assertTrue(result['pass'])
        changed=copy.deepcopy(record);changed['trace'][0]['observation_after']['memory']['prediction']['confidence']=.1
        with self.assertRaises(ValueError):audit.audit_trajectory(changed,.9,bert,qwen,{3},'reference',{'detector':2048,'actor':6144})

    def test_executed_actor_overflow_is_rejected_but_unexecuted_terminal_overflow_is_preserved(self):
        record,bert,qwen=self.context_fixture(actor_overflow=True)
        self.assertEqual(len(record['trace']),1)
        self.assertEqual(record['outcome']['reason'],'external_context_budget')
        self.assertTrue(audit.audit_trajectory(record,.9,bert,qwen,{3},'reference',{'detector':2048,'actor':6144})['pass'])
        with self.assertRaisesRegex(ValueError,'Executed actor context'):
            audit.audit_trajectory(record,.9,bert,qwen,{3},'reference',{'detector':2048,'actor':1})

    def test_complete_archive_contract_and_derived_joint_counts(self):
        # All model evidence below is explicitly synthetic; native token/world
        # replay is real, while pretrained-pin/layout dependencies are stubbed.
        for changed_counts in [False,True]:
            with self.subTest(changed_counts=changed_counts), tempfile.TemporaryDirectory() as directory:
                files,context,tokenizers,values,raw=self.fixture();root=Path(directory);source=root/'source'
                (source/'inputs').mkdir(parents=True);(source/'config').mkdir();transport=root/'transport';transport.mkdir()
                (transport/'TRANSPORT_MANIFEST.json').write_text('{"files":{}}')
                cfg=read(Path(__file__).resolve().parents[1]/'config/composite_controller_rank32_v2.json')
                (source/'config/composite_controller_rank32_v2.json').write_text(json.dumps(cfg))
                (source/'inputs/MANIFEST.json').write_text('{}')
                metadata=b'{}';metadata_sha=hashlib.sha256(metadata).hexdigest()
                pin={'files':[{'name':'fixture.json','size':2,'sha256':metadata_sha},
                    {'name':'model.safetensors','size':1,'sha256':'b'*64}]}
                (source/'inputs/MODERNBERT_PIN.json').write_text(json.dumps(pin))
                context.update(source_commit='a'*40,pin={'notebook':'farhad89/synthetic-only','transport_manifest_sha256':sha(transport/'TRANSPORT_MANIFEST.json')})
                context['evaluation_plan']['partition']=0
                context['model_pin']={'synthetic_fixture_only':True}
                proof=[{'name':'fixture.json','size':2,'sha256':metadata_sha},
                       {'name':'model.safetensors','size':1,'sha256':'b'*64}]
                loaded={'model_files':[],'loading_info':{'missing_keys':[],'mismatched_keys':[],
                    'error_msgs':[],'ignored_checkpoint_only_MTP_keys':[]}}
                values.update({'FROZEN_BASE_PACKAGE.json':{'plan':context['evaluation_plan'],
                    'backbone_instances':{'ModernBERT':1,'Qwen':1},'behavior_adapter':None,'trainable_parameters':0,
                    'full_validation_admitted':False,'actor':loaded,'detector_files':proof},
                    'FROZEN_LOADED_LAYOUT.json':loaded,'HARDWARE_PREFLIGHT.json':{},
                    'MODEL_FILE_VERIFICATION.json':[],'MODEL_METADATA_BYTES.json':{
                        'generation_config.json':base64.b64encode(b'{"eos_token_id":3}').decode()},
                    'DETECTOR_METADATA_BYTES.json':{'fixture.json':base64.b64encode(metadata).decode()},
                    'DETECTOR_FILE_VERIFICATION.json':proof})
                _,replayed=audit.replay(files,context,{},tokenizers,{3})
                count=replayed['expected_joint_forwards']+int(changed_counts);counts={'attempted':count,'completed':count}
                receipt={'source_commit':'a'*40,'stage':audit.MODE,'status':'complete','heldout_content_read':False,
                    'behavior_adapter_fitting':False,'optimizer_updates':0,'forward_counts':counts,'independent_joint_counts':counts}
                values.update({'RUN_RECEIPT.json':receipt,'PRECALL_FREEZE.json':{'source_commit':'a'*40,'mode':audit.MODE,
                    'pin':context['pin'],'input_manifest_sha256':sha(source/'inputs/MANIFEST.json'),
                    'bounds':audit.BOUNDS,'disk_policy':audit.DISK_POLICY},
                    'FORWARD_COUNTS.json':{**counts,'sealed_for_exit':True}})
                evidence=root/'raw';evidence.mkdir()
                for name,value in values.items():
                    target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value))
                for name,value in raw.items():
                    target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
                archive=root/'evidence.zip';sealed=make_evidence_archive(evidence,archive)
                controller={'source_commit':'a'*40,'mode':audit.MODE,'notebook':context['pin']['notebook'],
                    'child_state':{'child_exit_confirmed':True,'exit_code':0},'evidence':{'archive_sha256':sha(archive),'evidence_kind':'original_worker_export'},
                    'counts':{'counts_exact_at_exit':True,'forward_counts':counts},'complete_collection_exported':True,'cleanup':{'owned_cache_removed':True}}
                export={**receipt,**sealed}
                with patch.object(audit,'authorized_load_plan',return_value=context), \
                     patch.object(audit,'audit_runtime_metadata'),patch.object(audit,'parameter_audit',return_value={'synthetic_only':True}), \
                     patch.object(audit,'validate_download_proof'),patch.object(audit,'verify_metadata_bytes'), \
                     patch('src.composite_frozen_base_eval.selected_detector',return_value={}), \
                     patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizers[0]):
                    call=lambda:audit.audit_archive(archive,source,'a'*40,transport,controller,export)
                    if changed_counts:
                        with self.assertRaisesRegex(ValueError,'Unaccounted'):call()
                    else:
                        result=call();self.assertTrue(result['complete_base_partition_audited'])
                        self.assertFalse(result['full_system_validation_complete']);self.assertEqual(result['audit_model_forwards'],0)


if __name__=='__main__':unittest.main()
