import copy
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from src import composite_frozen_probe as probe
from src.composite_actor import Actor
from src.io_utils import digest
import test_composite_placement as placement_fixtures

ROOT = Path(__file__).resolve().parents[1]


class FrozenProbeTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'inputs').mkdir(); (root / 'config').mkdir()
        profile = (ROOT / probe.PROFILE_FILE).read_bytes()
        (root / probe.PROFILE_FILE).write_bytes(profile)
        model_pin = (ROOT / 'config/composite_qwen_file_pin_fc05.json').read_bytes()
        (root / 'config/composite_qwen_file_pin_fc05.json').write_bytes(model_pin)
        config = json.loads((ROOT / 'config/composite_controller_rank32_v2.json').read_text())['qwen']
        pin = {'platform': 'kaggle', 'account': 'farhad89', 'mode': probe.MODE,
               'source_commit': 'a' * 40, 'notebook': 'farhad89/frozen-probe-fixture',
               'original_scientific_input_manifest_sha256': probe.ORIGINAL_INPUTS,
               'transport_manifest_sha256': probe.TRANSPORT,
               'scientific_visible_devices': '0,1', **probe.BOUNDS}
        authorization = {'status': 'explicit_human_exception', 'authorized_by': 'human_user',
            'human_message_evidence_locator': 'synthetic-unit-test-only-not-cloud-authority',
            'mode': probe.MODE, 'scope': 'frozen_base_activation_analysis_only',
            'source_commit': 'a' * 40, 'notebook': pin['notebook'],
            'repository': config['repository'], 'revision': config['revision'],
            'profile_sha256': hashlib.sha256(profile).hexdigest(),
            'original_scientific_input_manifest_sha256': probe.ORIGINAL_INPUTS,
            'transport_manifest_sha256': probe.TRANSPORT,
            'scientific_visible_GPU_count': 2, 'maximum_new_stage_submissions': 1,
            'optimizer_updates': 0, 'behavior_adapter_fitting': False, 'heldout_content_read': False,
            'training_probe_conditions': probe.CONDITIONS, 'complete_prompts_no_truncation': True,
            **probe.BOUNDS}
        (root / 'inputs/QWEN_PIN.json').write_bytes(model_pin)
        (root / 'inputs' / probe.METADATA).write_text('{}')
        (root / 'RELEASE.json').write_text(json.dumps({'commit': 'a' * 40, 'files': {probe.PROFILE_FILE: hashlib.sha256(profile).hexdigest()}}))
        self.write(root, pin, authorization)
        return config, pin, authorization

    def write(self, root, pin, authorization):
        (root / 'inputs' / probe.PIN).write_text(json.dumps(pin))
        (root / 'inputs' / probe.AUTHORIZATION).write_text(json.dumps(authorization))
        (root / 'inputs/MANIFEST.json').write_text(json.dumps({file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in (root / 'inputs').iterdir() if file.name != 'MANIFEST.json'}))

    def test_proposal_and_wrong_exception_cannot_reach_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config, pin, authorization = self.fixture(root)
            for changes in [{'status': 'proposal_not_authorized'}, {'authorized_by': 'agent'},
                            {'scientific_visible_GPU_count': 1}, {'hard_timeout_seconds': 4500},
                            {'maximum_backbone_forwards': 401}, {'optimizer_updates': 1},
                            {'behavior_adapter_fitting': True}, {'heldout_content_read': True},
                            {'training_probe_conditions': ['own_pending']},
                            {'maximum_new_stage_submissions': True}, {'human_message_evidence_locator': None}]:
                with self.subTest(changes=changes):
                    self.write(root, pin, {**authorization, **changes})
                    with self.assertRaises(PermissionError):
                        probe.authorized_load_plan(root, 'a' * 40, config)
            (root / 'inputs' / probe.AUTHORIZATION).unlink()
            with self.assertRaises(PermissionError):
                probe.authorized_load_plan(root, 'a' * 40, config)

    def test_synthetic_authority_only_binds_metadata_and_not_runtime_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config, pin, authorization = self.fixture(root)
            with patch.object(probe, 'metadata_bounds', return_value={'device_weight_lower_bounds_bytes': [1, 2]}):
                context = probe.authorized_load_plan(root, 'a' * 40, config)
            self.assertTrue(context['load_attempt_authorized'])
            self.assertFalse(context['runtime_memory_admitted'])
            self.assertFalse(context['full_validation_admitted'])
            self.write(root, {**pin, 'maximum_backbone_forwards': 800}, authorization)
            with self.assertRaisesRegex(ValueError, 'bounds'):
                probe.authorized_load_plan(root, 'a' * 40, config)
            self.write(root, {**pin, 'transport_manifest_sha256': 'other'}, authorization)
            with self.assertRaisesRegex(ValueError, 'inputs'):
                probe.authorized_load_plan(root, 'a' * 40, config)

    def test_software_and_per_device_metadata_checks_do_not_claim_fit(self):
        packages = {'transformers':'5.19.0','bitsandbytes':'0.49.2','accelerate':'1.15.0',
                    'safetensors':'0.8.0','numpy':'2.2.2','psutil':'7.2.2'}
        probe.verify_software('2.11.0+cu128', packages)
        with self.assertRaises(RuntimeError):
            probe.verify_software('2.12.0', packages)
        with self.assertRaises(RuntimeError):
            probe.verify_software('2.11.0+cu128', {**packages, 'transformers':'other'})
        cuda = SimpleNamespace(is_available=lambda:True, device_count=lambda:2,
                               get_device_name=lambda index:'Tesla T4',
                               mem_get_info=lambda index:(15_000_000_000, 15_636_037_632))
        context = {'metadata_bounds': {'device_weight_lower_bounds_bytes':[8_743_381_400,8_749_178_080]}}
        with patch('src.composite_runtime.require_scientific_runtime'), patch('src.composite_runtime.platform', return_value='kaggle'), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0,1'}):
            self.assertFalse(probe.verify_hardware(context, cuda)['runtime_memory_admitted'])
            cuda.get_device_name = lambda index:'A100'
            with self.assertRaises(ValueError): probe.verify_hardware(context, cuda)
            cuda.get_device_name = lambda index:'Tesla T4'
            cuda.mem_get_info = lambda index:(100 if index == 1 else 15_000_000_000, 15_636_037_632)
            with self.assertRaises(MemoryError): probe.verify_hardware(context, cuda)
        with self.assertRaises(MemoryError): probe.guard_frozen_devices(cuda)

    def test_loaded_layout_checks_logical_nf4_shapes_and_actual_device_map(self):
        _, _, plan, _, _ = placement_fixtures.PlacementTests().fixture()
        names = ['model.language_model.layers.0.mlp.down_proj.weight', 'model.language_model.layers.18.mlp.down_proj.weight',
                 'model.language_model.embed_tokens.weight','lm_head.weight']
        params = {}; modules = {}
        for name,index in zip(names,[0,1,0,0]):
            device=SimpleNamespace(type='cuda',index=index); state=None
            if name in names[:2]:
                scale=SimpleNamespace(device=device)
                nested=SimpleNamespace(absmax=scale,code=scale,blocksize=256)
                state=SimpleNamespace(shape=[2,3],quant_type='nf4',nested=True,state2=nested,
                                      dtype=torch.bfloat16,blocksize=64,absmax=scale,code=scale)
            parameter=SimpleNamespace(requires_grad=False,device=device,shape=[3,1] if state else [2,3],
                                      quant_state=state,dtype=torch.uint8 if state else torch.bfloat16)
            params[name]=parameter
            modules[name.removesuffix('.weight')]=SimpleNamespace(weight=parameter,compute_dtype=torch.bfloat16,
                                                                 in_features=3,out_features=2)
        model = SimpleNamespace(named_parameters=lambda:iter(params.items()), hf_device_map=plan['device_map'],
                                get_submodule=lambda name:modules[name])
        context = {'plan':plan,'metadata':{'tensors':{name:{'shape':[2,3]} for name in names}},
                   'metadata_bounds':{'runtime_parameter_names':names}}
        result=probe.verify_loaded_layout(model,context)
        self.assertEqual(result['original_runtime_parameters_verified'],4)
        self.assertEqual(len(result['NF4_double_quant_projections']),2)
        self.assertEqual(len(result['unquantized_BF16']),2)
        params[names[1]].device.index = 0
        with self.assertRaises(ValueError): probe.verify_loaded_layout(model, context)
        params[names[1]].device.index = 1
        params['lm_head.weight'].dtype=torch.float32
        with self.assertRaisesRegex(ValueError,'BF16'): probe.verify_loaded_layout(model,context)
        params['lm_head.weight'].dtype=torch.bfloat16
        params[names[0]].quant_state.quant_type='fp4'
        with self.assertRaisesRegex(ValueError,'NF4'): probe.verify_loaded_layout(model,context)
        params[names[0]].quant_state.quant_type='nf4';params[names[0]].quant_state.nested=False
        with self.assertRaisesRegex(ValueError,'double'): probe.verify_loaded_layout(model,context)
        params[names[0]].quant_state.nested=True
        model.hf_device_map = {**plan['device_map'], 'lm_head':'cpu'}
        with self.assertRaises(ValueError): probe.verify_loaded_layout(model, context)

    def test_original_pin_and_fixed_config_reject_mutually_consistent_mutation(self):
        original=json.loads((ROOT/'config/composite_qwen_file_pin_fc05.json').read_text())
        self.assertTrue(probe.verify_original_model_pin(original))
        changed=copy.deepcopy(original);changed['files'][0]['size']+=1
        with self.assertRaises(ValueError):probe.verify_original_model_pin(changed)
        cfg=json.loads((ROOT/'config/composite_controller_rank32_v2.json').read_text())['qwen']
        profile=json.loads((ROOT/probe.PROFILE_FILE).read_text());profile['model_config_sha256']='e'*64
        with self.assertRaises(ValueError):probe.candidate_plan(cfg,profile)
        proofs=[]
        fixed={'config.json':probe.CONFIG_SHA256,'model.safetensors.index.json':probe.INDEX_SHA256}
        for item in original['files']:
            proofs.append({'name':item['name'],'size':item['size'],'sha256':fixed.get(item['name']) or item.get('sha256') or 'c'*64})
        probe.validate_download_proof({'model_pin':original},proofs)
        next(value for value in proofs if value['name']=='config.json')['sha256']='0'*64
        with self.assertRaises(ValueError):probe.validate_download_proof({'model_pin':original},proofs)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'config.json').write_text('{"mutated":true}')
            with patch('src.composite_store.verify_model_files',return_value=proofs):
                with self.assertRaisesRegex(ValueError,'config'):
                    probe.verify_downloaded_model({'model_pin':original},path)

    def test_raw_metadata_bytes_bind_sha_and_original_git_blob_without_weights(self):
        raw=b'{"source":"synthetic unit fixture"}';name='fixture.json'
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        pin={'files':[{'name':name,'size':len(raw),'sha256':None,'git_blob_id':blob}]}
        proofs=[{'name':name,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]
        encoded={name:base64.b64encode(raw).decode()}
        result=probe.verify_metadata_bytes(pin,proofs,encoded)
        self.assertEqual(result,{'metadata_files_verified':1,'pretrained_weight_bytes_exported':0})
        altered=raw.replace(b'fixture',b'altered')
        changed={name:base64.b64encode(altered).decode()}
        with self.assertRaises(ValueError):probe.verify_metadata_bytes(pin,proofs,changed)
        changed_proof=[{**proofs[0],'sha256':hashlib.sha256(altered).hexdigest()}]
        with self.assertRaisesRegex(ValueError,'Git blob'):probe.verify_metadata_bytes(pin,changed_proof,changed)
        with self.assertRaises(ValueError):probe.verify_metadata_bytes(pin,proofs,{**encoded,'model.safetensors':'AA=='})

    def test_cache_namespace_and_complete_pretrained_loading_fail_closed(self):
        commit='a'*40
        good=Path('/kaggle/temp/research4_frozen_probe_cache')/commit/'run-one'
        self.assertEqual(probe.protected_cache_path(good,commit),good.resolve())
        for cache in ['/kaggle/working/weights','/Research3/cache',good.parent]:
            with self.assertRaises(ValueError): probe.protected_cache_path(cache,commit)
        probe.validate_loading_info({'missing_keys':set(),'unexpected_keys':{'mtp.layers.0.weight'},'mismatched_keys':set(),'error_msgs':[]})
        for info in [{'missing_keys':{'lm_head.weight'}},{'unexpected_keys':{'another_model.weight'}},
                     {'mismatched_keys':{'weight'}},{'error_msgs':['conversion failed']}]:
            with self.assertRaises(ValueError): probe.validate_loading_info(info)

    def test_capture_is_observational_complete_and_removes_hooks(self):
        layers = [nn.Identity(), nn.Identity()]
        value = torch.tensor([[[1.,2.,3.],[4.,5.,6.]]], requires_grad=True)
        original = value
        with probe.LastTokenCapture(layers,3,2) as captured:
            for layer in layers: value = layer(value)
        self.assertIs(value, original)
        self.assertEqual([item['values'] for item in captured.complete()], [[4.,5.,6.]]*2)
        self.assertTrue(all(not layer._forward_hooks for layer in layers))
        self.assertIsNone(original.grad)
        with self.assertRaises(ValueError):
            with probe.LastTokenCapture(layers,3,3): layers[0](value)
        self.assertTrue(all(not layer._forward_hooks for layer in layers))
        with self.assertRaises(ValueError):
            with probe.LastTokenCapture(layers,3,2):
                layers[0](value); layers[0](value)
        self.assertTrue(all(not layer._forward_hooks for layer in layers))

    def test_training_probe_selection_keeps_original_order_and_rejects_validation(self):
        row = {'split':'training','state_id':'q','case_id':'q','sequence':0,
               'condition':'quiet_pending','terminal':False,'text':'quiet'}
        own = {**row,'state_id':'o','case_id':'o','condition':'own_pending','text':'own'}
        later = {**row,'state_id':'later','case_id':'later'}
        self.assertEqual(probe.select_training_probes([row,later,own]),[row,own])
        with self.assertRaises(PermissionError): probe.select_training_probes([row,{**own,'split':'validation'}])
        with self.assertRaises(ValueError): probe.select_training_probes([row])

    def actor_stub(self):
        actor = Actor.__new__(Actor); nn.Module.__init__(actor)
        actor.frozen_analysis_only = True; actor.frozen_probe_counts = {'attempted':0,'completed':0}
        actor.cfg = {'maximum_new_tokens':160}; actor.generating = False; actor.adapters = {}
        actor.lock = threading.RLock(); actor.model = nn.Module(); actor.model._cache = None
        actor.guard = lambda:None; actor.on_forward = None; actor._last_cache = None
        return actor

    def test_frozen_actor_rejects_fitting_adapters_shortcuts_and_kernel_changes(self):
        actor = self.actor_stub()
        operations = [lambda:actor.configure({'scope':'all_blocks'}), lambda:actor.training_context(),
                      lambda:actor.restore({}), lambda:actor.supervised_loss({}),
                      lambda:actor.bypass_check([]), lambda:actor.select_kernels('fla'),
                      lambda:actor.generate('fixture',True), lambda:actor.generate('fixture',False,2)]
        for index, operation in enumerate(operations):
            with self.subTest(index=index), self.assertRaises(PermissionError): operation()
        self.assertEqual(actor.frozen_probe_counts, {'attempted':0,'completed':0})

    def test_frozen_cache_provenance_and_forward_ceiling(self):
        class Cache:
            def get_seq_length(self): return 0
        actor = self.actor_stub(); previous = Cache(); fresh = Cache()
        actor.generating = True; actor._last_cache = weakref.ref(previous); actor._trace = {'prefill_checked':False}
        with patch.object(probe,'guard_frozen_devices'):
            actor._before(None,(),{'past_key_values':fresh})
            actor._after(None,(),SimpleNamespace(logits=torch.zeros(1,1,3)))
            self.assertEqual(actor.frozen_probe_counts, {'attempted':1,'completed':1})
            self.assertTrue(actor._trace['previous_cache_alive'])
            self.assertFalse(actor._trace['previous_cache_reused'])
            actor._trace = {'prefill_checked':False}
            with self.assertRaisesRegex(RuntimeError,'reused'): actor._before(None,(),{'past_key_values':fresh})
            actor.generating = False; actor.frozen_probe_counts['attempted'] = 400
            with self.assertRaises(TimeoutError): actor._before(None,(),{})
            self.assertEqual(actor.frozen_probe_counts['attempted'],400)

    def raw_fixture(self):
        _, _, plan, metadata, base = placement_fixtures.PlacementTests().fixture()
        rows = [{'split':'training','state_id':'q','case_id':'q','sequence':0,'condition':'quiet_pending','terminal':False,'text':'quiet'},
                {'split':'training','state_id':'o','case_id':'o','sequence':0,'condition':'own_pending','terminal':False,'text':'own'}]
        rows += [{'split':'training','state_id':'filler-'+str(i),'case_id':'case-'+str(i),'sequence':0,
                  'condition':'peer_pending','terminal':False,'text':'synthetic filler'} for i in range(718)]
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs): return [0,1] if messages[-1]['content']=='quiet' else [1,2]
            def decode(self,ids,**kwargs): return '<tool_call>\n<function=catalog>\n</function>\n</tool_call>'
        tokenizer = Tokenizer(); raw = tokenizer.decode([2])
        record = {key:value for key,value in base.items() if key not in ['input_token_ids','last_token_activations','cache_boundary']}
        model_pin=json.loads((ROOT/'config/composite_qwen_file_pin_fc05.json').read_text())
        proofs=[];fixed={'config.json':probe.CONFIG_SHA256,'model.safetensors.index.json':probe.INDEX_SHA256}
        for item in model_pin['files']:
            proofs.append({'name':item['name'],'size':item['size'],'sha256':fixed.get(item['name']) or item.get('sha256') or 'c'*64})
        for name,value in record['parameters_before'].items():
            value['dtype']='torch.uint8' if probe.text_projection(name) else ('torch.float32' if metadata['tensors'][name]['dtype']=='F32' else 'torch.bfloat16')
        record['parameters_after']=copy.deepcopy(record['parameters_before'])
        quantized={};unquantized={}
        for name,value in record['parameters_before'].items():
            device=value['device'];shape=value['logical_shape']
            if name.endswith('embed_tokens.weight') or name=='lm_head.weight':
                unquantized[name]={'dtype':'torch.bfloat16','shape':shape,'device':device}
            elif probe.text_projection(name):
                quantized[name]={'quant_type':'nf4','double_quantization':True,'storage_dtype':'torch.uint8',
                    'compute_dtype':'torch.bfloat16','state_dtype':'torch.bfloat16','blocksize':64,'nested_blocksize':256,
                    'logical_shape':shape,'state_devices':{key:device for key in ['absmax','code','nested_absmax','nested_code']}}
        layout={'original_runtime_parameters_verified':len(record['parameters_before']),'model_instances':1,
                'optimizer_updates':0,'device_map':plan['device_map'],'unquantized_BF16':unquantized,
                'NF4_double_quant_projections':quantized,'runtime_memory_admitted':False}
        record.update(source_commit='a'*40,authorization_sha256='d'*64,probe_conditions=probe.CONDITIONS,
                      behavior_adapter_fitting=False,
                      simulator_actions_executed=0,forward_counts={'attempted':6,'completed':6},
                      math_profile={'cuda_matmul_allow_tf32':False,'caller_autocast_enabled':False},
                      loading_info={'missing_keys':[],'mismatched_keys':[],'error_msgs':[],'ignored_checkpoint_only_MTP_keys':[]},
                      model_file_verification=proofs,loaded_layout=layout,
                      model_metadata_bytes={},
                      data_provenance={'source_training_observations':720,'training_rows_canonical_sha256':digest(rows),'heldout_content_read':False},
                      device_memory=[{'device':i,'free_bytes':2*1024**3,'total_bytes':15*1024**3,
                                      'allocated_bytes':1024,'reserved_bytes':2048,'peak_allocated_bytes':1024,'peak_reserved_bytes':2048} for i in [0,1]],probes=[])
        for index,row in enumerate(rows[:2]):
            ids = [0,1] if index==0 else [1,2]
            cache = {'prefill_checked':True,'initial_cache_length':0,'previous_cache_reused':False,
                     'cache_object_present':True,'cache_identity':index+10,'previous_cache_alive':index==1,
                     'previous_cache_identity':10 if index==1 else None}
            record['probes'].append({'state_id':row['state_id'],'case_id':row['case_id'],'condition':row['condition'],
                'input_token_ids':ids,'prompt_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                'last_token_activations':copy.deepcopy(base['last_token_activations']), 'raw_last_logits':[0.,0.,1.,0.],
                'cache_boundary':{'prefill_checked':True,'initial_cache_length':0,'reused_previous_response_cache':False},
                'generation':{'input_token_ids':ids,'generated_token_ids':[2,3],'raw_text':raw,
                              'actions':[{'tool':'catalog','arguments':{}}],'finished_with_eos':True,
                              'adapter_enabled':False,'kernel_profile':'reference','cache':cache,
                              'generation_parameters':{'max_new_tokens':160,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}}})
        context = {'source_commit':'a'*40,'authorization_sha256':'d'*64,'plan':plan,'metadata':metadata,
                   'model_pin':model_pin,'metadata_bounds':probe.metadata_bounds(metadata,plan)}
        return record,rows,context,tokenizer

    def test_native_raw_audit_replays_both_complete_probes_without_model_forwards(self):
        record,rows,context,tokenizer = self.raw_fixture()
        audit = self.audit_raw_fixture(record,rows,context,tokenizer)
        self.assertEqual(audit['probes'],2); self.assertEqual(audit['native_parse_successes'],2)
        self.assertFalse(audit['full_validation_admitted'])
        self.assertEqual(audit['audit_model_forwards'],0)

    def test_raw_audit_rejects_truncated_logits_changed_tokens_cache_and_source(self):
        record,rows,context,tokenizer = self.raw_fixture()
        for kind in ['logits','tokens','cache','source','budget','accounting']:
            value = copy.deepcopy(record)
            if kind=='logits': value['probes'][0]['raw_last_logits']=[0.]
            elif kind=='tokens': value['probes'][0]['generation']['raw_text']='fabricated'
            elif kind=='cache': value['probes'][1]['generation']['cache']['cache_identity']=10
            elif kind=='source': value['source_commit']='b'*40
            elif kind=='budget': value['probes'][0]['generation']['generation_parameters']['max_new_tokens']=2
            else: value['forward_counts']['completed']=5
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit_raw_fixture(value,rows,context,tokenizer)

    def test_raw_audit_rejects_early_eos_early_stop_and_divergent_cache_records(self):
        record,rows,context,tokenizer=self.raw_fixture()
        for kind in ['eos_then_more','short_without_eos','raw_prefill','raw_length','raw_reuse','unknown_reuse','unknown_heldout']:
            value=copy.deepcopy(record);generation=value['probes'][0]['generation']
            if kind=='eos_then_more':generation['generated_token_ids']=[3,2,3];value['forward_counts']={'attempted':7,'completed':7}
            elif kind=='short_without_eos':generation.update(generated_token_ids=[2],finished_with_eos=False,actions=None);value['forward_counts']={'attempted':5,'completed':5}
            elif kind=='raw_prefill':generation['cache']['prefill_checked']=False
            elif kind=='raw_length':generation['cache']['initial_cache_length']=1
            elif kind=='raw_reuse':generation['cache']['previous_cache_reused']=True
            elif kind=='unknown_reuse':generation['cache']['previous_cache_reused']=None
            else:value['heldout_content_read']=None
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.audit_raw_fixture(value,rows,context,tokenizer)

    def test_raw_quantization_and_pin_tampering_are_not_mutually_authorizing(self):
        record,rows,context,tokenizer=self.raw_fixture()
        for kind in ['head_dtype','double_quant','weight_pin']:
            value=copy.deepcopy(record);ctx=copy.deepcopy(context)
            if kind=='head_dtype':
                value['parameters_before']['lm_head.weight']['dtype']='torch.float32';value['parameters_after']['lm_head.weight']['dtype']='torch.float32'
            elif kind=='double_quant':
                first=next(iter(value['loaded_layout']['NF4_double_quant_projections']))
                value['loaded_layout']['NF4_double_quant_projections'][first]['double_quantization']=False
            else:
                ctx['model_pin']['files'][0]['size']+=1;value['model_file_verification'][0]['size']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.audit_raw_fixture(value,rows,ctx,tokenizer)

    def audit_raw_fixture(self,record,rows,context,tokenizer):
        # Raw generation/layout fixtures don't vendor a large real tokenizer.
        # Actual byte/SHA/Git-blob checking has its own tiny fixture above.
        with patch.object(probe,'verify_metadata_bytes',return_value={'metadata_files_verified':6,'pretrained_weight_bytes_exported':0}):
            return probe.audit_probe_record(record,rows,context,tokenizer,{3})


if __name__=='__main__': unittest.main()
