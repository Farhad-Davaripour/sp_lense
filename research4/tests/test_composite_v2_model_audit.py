from copy import deepcopy
import hashlib,io,json,unittest,zipfile
import torch
from src.io_utils import digest
from src.composite_training import tensor_digest
from src.composite_generalized_interface import PROFILE
from src.composite_v2_model_audit import (EvidenceFiles,ModelEvidence,audit_model_files,
    checkpoint_evidence,frozen_integrity,require_qualification_evidence,provider_closure,ProviderClosure,
    audit_optimizer_bytes,audit_runtime,audit_bypass)


def fixture_files():
    cfg={'unit_fixture':True};projection='layers.0.mlp.up_proj'
    weights={projection+'.A':torch.ones(32,2),projection+'.B':torch.ones(2,32)}
    stream=io.BytesIO();torch.save({'config':cfg,'weights':weights,'placement':{'name':'all_blocks','scope':'all_blocks'},
                                 'native_interface_profile':PROFILE},stream);checkpoint=stream.getvalue()
    stage={'phase':'qualification','data_release_sha256':'d'*64}
    token_plan={'qualification_state_ids':['a','b'],'records':[{'state_id':'a'},{'state_id':'b'}]}
    memory={'device':0,'free_bytes':1024**3,'total_bytes':2*1024**3,'allocated_bytes':1024,
            'reserved_bytes':2048,'peak_allocated_bytes':2048,'peak_reserved_bytes':4096}
    journal={'optimizer_steps_attempted':1,'optimizer_steps_completed':1,'backward_microsteps_completed':2,
             'layer_forward_invocations':[2]*64,'backward_recomputation_invocations':[2]*64,'in_backward':False}
    records={'RUN_RECEIPT.json':{'source_commit':'a'*40,'phase':'qualification','native_interface_profile':PROFILE,
              'public_fixture_test_mode':False,'status':'complete','heldout_content_read':False,
              'forward_counts':{'attempted':2,'completed':2},'counts_exact_at_controlled_exit':True},
        'PRECALL_FREEZE.json':{'stage':stage,'workspace':{'fixture':True},'data_release_sha256':'d'*64},
        'TRAINING_TOKEN_PLAN.json':token_plan,
        'BEHAVIOR_TRAINING.json':{'optimizer_updates':1,'sampled_state_ids':['a','b'],'losses':[1.,2.],
          'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),'heldout_content_read':False},
        'ADAPTER_INITIAL_HASHES.json':{n:tensor_digest(torch.zeros_like(t)) for n,t in weights.items()},
        'TRAINING_TERMINAL_JOURNAL.json':journal,'training/update_001.json':{**journal,'memory':memory}}
    for i,state in enumerate(['a','b']):
        records['training/backward_'+str(i+1).zfill(4)+'.json']={'update':1,'microstep':i,'state_id':state,
            'loss':float(i+1),'memory':memory,'gradients':[{'name':n,'device':'cuda:0','dtype':'torch.float32','norm':1.} for n in weights]}
    class Files:
        sha256='f'*64
        def __init__(self):self.records=deepcopy(records);self.raw={'behavior.pt':checkpoint}
        @property
        def manifest(self):return {n:'0'*64 for n in list(self.records)+list(self.raw)}
        def json(self,name):return deepcopy(self.records[name])
        def get(self,name):return self.raw[name]
    context={'stage':stage,'source_commit':'a'*40,'workspace':{'fixture':True},'configuration':{'qwen':cfg},
             'original_shapes':{'base':[2,2]},'projections':{projection:[2,2]}}
    return Files(),context,token_plan


class ModelAuditTests(unittest.TestCase):
    def test_tiny_complete_recorded_counts_never_qualify_a_real_stage(self):
        files,context,tokens=fixture_files()
        proof=audit_model_files(files,context,tokens,public_fixture=True).receipt()
        self.assertEqual(proof['recorded_update_count_verified'],1)
        self.assertEqual(proof['checkpoint']['adapter_parameters'],128)
        self.assertFalse(proof['scientific_qualification_admitted']);self.assertFalse(proof['model_tensor_integrity_audited'])
        self.assertIn('live_backbone_bytes_and_flags_not_exported',proof['missing_or_unverified'])
        self.assertIsNone(proof['owned_compute_released'])
        self.assertFalse(proof['absence_of_unlogged_calls_proved'])

    def test_fake_worker_success_and_caller_boolean_do_not_create_a_permit(self):
        files,context,tokens=fixture_files();value=audit_model_files(files,context,tokens,public_fixture=True)
        with self.assertRaises(PermissionError):require_qualification_evidence(value,{'complete_training_qualification_audited':True})
        with self.assertRaises(PermissionError):require_qualification_evidence({'scientific_qualification_admitted':True},{})
        with self.assertRaises(PermissionError):ModelEvidence({'scientific_qualification_admitted':True})
        from src.composite_v2_model_audit import EvaluationIntegrity,read_evaluation_integrity,bound_evaluation_integrity
        with self.assertRaises(PermissionError):EvaluationIntegrity({'complete_model_integrity_audited':True})
        with self.assertRaises(PermissionError):read_evaluation_integrity(files,'.','a'*40,{}, {},value)
        self.assertFalse(bound_evaluation_integrity(True,'a'*40,'b'*64,{}))

    def test_missing_exports_and_failed_counts_remain_unknown(self):
        files,context,tokens=fixture_files();del files.records['BEHAVIOR_TRAINING.json']
        files.records['RUN_RECEIPT.json']['status']='failed'
        proof=audit_model_files(files,context,tokens,public_fixture=True).receipt()
        self.assertIsNone(proof['recorded_update_count_verified'])
        self.assertIn('BEHAVIOR_TRAINING.json',proof['missing_raw_files'])
        self.assertFalse(proof['complete_training_qualification_audited'])

    def test_changed_samples_losses_gradient_coverage_memory_and_counts_fail(self):
        for kind in ['sample','gradient','loss','memory','counts','boolean_count']:
            files,context,tokens=fixture_files()
            if kind=='sample':files.records['BEHAVIOR_TRAINING.json']['sampled_state_ids']=['foreign','b']
            elif kind=='gradient':files.records['training/backward_0001.json']['gradients'].pop()
            elif kind=='loss':files.records['BEHAVIOR_TRAINING.json']['losses'][0]=float('nan')
            elif kind=='memory':files.records['training/backward_0001.json']['memory']['free_bytes']=0
            elif kind=='counts':files.records['RUN_RECEIPT.json']['forward_counts']['completed']=0
            else:files.records['TRAINING_TERMINAL_JOURNAL.json']['optimizer_steps_completed']=True
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):audit_model_files(files,context,tokens,public_fixture=True)

    def test_checkpoint_bytes_shapes_dtype_and_tensor_coverage_are_independent(self):
        files,context,_=fixture_files();training=files.records['BEHAVIOR_TRAINING.json'];initial=files.records['ADAPTER_INITIAL_HASHES.json']
        with self.assertRaises(ValueError):checkpoint_evidence(b'changed',training['checkpoint_sha256'],context['configuration']['qwen'],context['projections'],initial)
        for kind in ['shape','dtype','coverage']:
            saved=torch.load(io.BytesIO(files.get('behavior.pt')),weights_only=True)
            name=next(iter(saved['weights']))
            if kind=='shape':saved['weights'][name]=torch.ones(1,1)
            elif kind=='dtype':saved['weights'][name]=saved['weights'][name].double()
            else:saved['weights'].pop(name)
            stream=io.BytesIO();torch.save(saved,stream);raw=stream.getvalue()
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):checkpoint_evidence(raw,hashlib.sha256(raw).hexdigest(),context['configuration']['qwen'],context['projections'],initial)

    def test_cached_and_incomplete_or_changed_live_backbone_cannot_pass(self):
        proof={'live_recomputed':True,'original_backbone_only':True,'auxiliary_sha256':'b'*64,
            'parameters':[{'name':'base','logical_shape':[2,2],'shape':[2,2],'dtype':'torch.float32',
                           'requires_grad':False,'gradient_is_none':True,'device':'cuda:0','sha256':'a'*64}]}
        self.assertTrue(frozen_integrity(proof,deepcopy(proof),{'base':[2,2]})['live_bytes_recomputed_and_equal'])
        for field,value in [('live_recomputed',False),('parameters',[]),('auxiliary_sha256','c'*64)]:
            changed=deepcopy(proof);changed[field]=value
            with self.assertRaises(ValueError):frozen_integrity(proof,changed,{'base':[2,2]})
        changed=deepcopy(proof);changed['parameters'][0]['gradient_is_none']=False
        with self.assertRaises(ValueError):frozen_integrity(proof,changed,{'base':[2,2]})

    def test_archive_checks_hash_paths_duplicate_json_and_sealed_names(self):
        def archive(files,manifest=None):
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as stream:
                for name,raw in files.items():stream.writestr(name,raw)
                stream.writestr('MANIFEST.json',json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()}) if manifest is None else manifest)
            return out.getvalue()
        raw=archive({'RUN_RECEIPT.json':b'{}'})
        reader=EvidenceFiles(raw,hashlib.sha256(raw).hexdigest());self.assertEqual(reader.json('RUN_RECEIPT.json'),{});reader.close()
        for raw in [archive({'../escape':b'{}'}),archive({'sealed/secret.json':b'{}'}),
                    archive({'x':b'{}'},'{"x":"a","x":"b"}')]:
            with self.assertRaises((ValueError,PermissionError)):EvidenceFiles(raw,hashlib.sha256(raw).hexdigest())

    def test_provider_release_needs_both_exact_terminal_run_responses(self):
        host='https://adb-7405616135490449.9.azuredatabricks.net'
        air={'run_id':'123','status':'SUCCESS'}
        jobs={'run_id':123,'state':{'life_cycle_state':'TERMINATED','result_state':'SUCCESS'}}
        value=provider_closure(host,123,'a'*64,air,jobs)
        self.assertTrue(value.receipt['owned_compute_released'])
        with self.assertRaises(PermissionError):ProviderClosure({'owned_compute_released':True})
        for bad_air,bad_jobs in [({**air,'status':'RUNNING'},jobs),(air,{**jobs,'run_id':456}),
            (air,{'run_id':123,'state':{'life_cycle_state':'TERMINATED','result_state':'FAILED'}})]:
            with self.assertRaises(ValueError):provider_closure(host,123,'a'*64,bad_air,bad_jobs)

    def test_raw_AdamW_bytes_are_checked_without_constructing_an_optimizer(self):
        moments={'p':{'step':torch.tensor(1.),'exp_avg':torch.zeros(2,2),'exp_avg_sq':torch.ones(2,2)}}
        def raw(value):
            stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()
        self.assertEqual(audit_optimizer_bytes(raw(moments),{'p':[2,2]},1)['raw_AdamW_tensors_verified'],2)
        for field,value in [('step',torch.tensor(2.)),('exp_avg',torch.zeros(1,1)),('exp_avg_sq',-torch.ones(2,2))]:
            bad=deepcopy(moments);bad['p'][field]=value
            with self.assertRaises(ValueError):audit_optimizer_bytes(raw(bad),{'p':[2,2]},1)

    def test_raw_bypass_proof_rejects_truncated_vocabulary_and_changed_OFF(self):
        import numpy as np
        files,_,tokens=fixture_files()
        for row in tokens['records']:row['input_token_ids']=[1,2]
        manifest={'logit_shape':[1,1,3],'records':[]}
        for i in range(2):
            row={'input_token_ids':[1,2]}
            for kind in ['wrapped_OFF','bare','wrapped_ON']:
                out=io.BytesIO();np.save(out,np.zeros((1,1,3),dtype=np.float32)+(kind=='wrapped_ON'))
                path='bypass/'+str(i).zfill(2)+'_'+kind+'.npy';files.raw[path]=out.getvalue()
                row[kind]={'path':path,'sha256':hashlib.sha256(out.getvalue()).hexdigest()}
            manifest['records'].append(row)
        files.records['BYPASS_RAW_MANIFEST.json']=manifest
        with self.assertRaisesRegex(ValueError,'vocabulary'):audit_bypass(files,tokens,{'vocabulary':4})
        bad=deepcopy(manifest);bad['records'][0]['bare']['sha256']='f'*64
        files.records['BYPASS_RAW_MANIFEST.json']=bad
        with self.assertRaisesRegex(ValueError,'changed'):audit_bypass(files,tokens,{'vocabulary':3})

    def test_runtime_reader_requires_exact_software_and_quantization_evidence(self):
        from src.composite_frozen_probe import text_projection
        name='model.language_model.layers.0.mlp.up_proj.weight'
        self.assertTrue(text_projection(name))
        quant={'quant_type':'nf4','double_quantization':True,'storage_dtype':'torch.uint8',
            'compute_dtype':'torch.bfloat16','state_dtype':'torch.bfloat16','blocksize':64,'nested_blocksize':256,
            'logical_shape':[2,2],'state_devices':{k:'cuda:0' for k in ['absmax','code','nested_absmax','nested_code']}}
        runtime={'torch':'2.11.0+cu128','packages':{'transformers':'5.19.0','bitsandbytes':'0.49.2',
            'accelerate':'1.15.0','safetensors':'0.8.0','numpy':'2.2.2','psutil':'7.2.2'},
            'visible_GPUs':1,'BF16_supported':True,'TF32_enabled':False,'device_map':{'':0},
            'kernel_profile':'reference','full_attention':'sdpa','quantized_original_projections':{name:quant}}
        live={'parameters':[{'name':name,'dtype':'torch.uint8'}]}
        self.assertTrue(audit_runtime(runtime,live,{name:[2,2]})['software_and_original_quantization_verified'])
        changed=deepcopy(runtime);changed['packages']['transformers']='changed'
        with self.assertRaises(RuntimeError):audit_runtime(changed,live,{name:[2,2]})
        changed=deepcopy(runtime);changed['quantized_original_projections'][name]['double_quantization']=False
        with self.assertRaises(ValueError):audit_runtime(changed,live,{name:[2,2]})

    def test_live_capture_recomputes_tiny_tensor_bytes_instead_of_cached_hashes(self):
        from types import SimpleNamespace
        from src.composite_v2_model_audit import capture_live_backbone
        parameter=torch.nn.Parameter(torch.ones(2,2),requires_grad=False)
        actor=SimpleNamespace(base_references=[('base',parameter)],base_buffers=[])
        first=capture_live_backbone(actor)
        with torch.no_grad():parameter.add_(1)
        second=capture_live_backbone(actor)
        self.assertTrue(first['live_recomputed'])
        self.assertNotEqual(first['parameters'][0]['sha256'],second['parameters'][0]['sha256'])


if __name__=='__main__':unittest.main()
