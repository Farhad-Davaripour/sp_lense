"""Read-only model export checks; no pretrained model construction or forwards.

Incomplete legacy worker exports produce explicit blockers, not qualification.
Hash equality alone never proves a live post-fit recomputation or absence of
unlogged calls. Public/tiny fixtures cannot produce a scientific closeout.
"""
from copy import deepcopy
import hashlib,io,json,math,re,zipfile
from .io_utils import digest
from .composite_generalized_interface import PROFILE

READER='v2-model-raw-export-audit-v1'
_KEY=object()


def _same(actual,expected,message):
    if digest(actual)!=digest(expected):raise ValueError(message)


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Missing exact evidence digest')


class ModelEvidence:
    def __init__(self,receipt,key=None):
        if key is not _KEY:raise PermissionError('Read the raw model export')
        self._value=deepcopy(receipt);self._key=key
    def receipt(self):return deepcopy(self._value)


class EvaluationIntegrity:
    def __init__(self,receipt,key=None):
        if key is not _KEY:raise PermissionError('Read the exact evaluation model evidence')
        self._value=deepcopy(receipt);self._key=key
    def receipt(self):return deepcopy(self._value)


class ProviderClosure:
    def __init__(self,receipt,key=None):
        if key is not _KEY:raise PermissionError('Reconcile the exact provider responses')
        self.receipt=deepcopy(receipt);self._key=key


def provider_closure(host,run_id,archive_sha256,air,jobs):
    """Read saved API responses, never query/cancel a job or infer from emptiness.

    Transport/host selection and ownership come from the external submission
    reconciler. Both exact run IDs and terminal-success responses are required.
    No raw `released=True` boolean is an accepted substitute.
    """
    _hash(archive_sha256)
    from .workspace_accounts import profiles
    if host not in {value['host'] for value in profiles()['accounts'].values()}:
        raise ValueError('Exact explicitly configured Databricks workspace host required')
    if type(run_id) is not int or run_id<=0 or type(jobs.get('run_id')) is not int or jobs['run_id']!=run_id:
        raise ValueError('Exact provider run identity required')
    if str(air.get('run_id'))!=str(run_id) or air.get('status') not in ['SUCCESS','SUCCEEDED']:
        raise ValueError('AIR terminal success not verified')
    state=jobs.get('state',{})
    if state.get('life_cycle_state')!='TERMINATED' or state.get('result_state')!='SUCCESS':
        raise ValueError('Jobs terminal success not verified')
    return ProviderClosure({'host':host,'run_id':run_id,'raw_archive_sha256':archive_sha256,
        'AIR_response_sha256':digest(air),'Jobs_response_sha256':digest(jobs),'owned_compute_released':True},_KEY)


class EvidenceFiles:
    """Bounded checksum-verified archive, never extracted or executed."""
    def __init__(self,raw,sha256,maximum_bytes=1024**3):
        if hashlib.sha256(raw).hexdigest()!=sha256:raise ValueError('Model archive SHA differs')
        self.sha256=sha256;self.stream=zipfile.ZipFile(io.BytesIO(raw))
        infos=self.stream.infolist();names=[v.filename for v in infos]
        if type(maximum_bytes) is not int or not 0<maximum_bytes<=4*1024**3:raise ValueError('Reviewed raw export envelope required')
        if len(names)!=len(set(names)) or sum(v.file_size for v in infos)>maximum_bytes:
            raise ValueError('Duplicate members or unchanged raw export bound exceeded')
        if any(not n or n.startswith('/') or any(p in ['','..','.'] for p in n.split('/'))
               or '\\' in n or ':' in n or 'sealed' in n.lower() or 'heldout' in n.lower() for n in names):
            raise PermissionError('Unsafe or held-out evidence member')
        if any(v.is_dir() or v.flag_bits&1 or (v.external_attr>>16)&0o170000==0o120000 for v in infos):
            raise ValueError('Only ordinary unencrypted evidence files are allowed')
        self.manifest=self.json('MANIFEST.json')
        if set(names)!=set(self.manifest)|{'MANIFEST.json'}:raise ValueError('Unaccounted raw evidence member')
        for name,expected in self.manifest.items():
            _hash(expected)
            if hashlib.sha256(self.get(name)).hexdigest()!=expected:raise ValueError('Raw evidence member changed')
    def get(self,name):return self.stream.read(name)
    def json(self,name):
        from .native27_tools import unique_json
        return unique_json(self.get(name).decode('utf-8'))
    def close(self):self.stream.close()


def capture_live_backbone(actor):
    """Reusable future export hook; reads live tensors, performs no model call."""
    from .composite_training import tensor_digest,auxiliary_digest
    from .composite_frozen_probe import logical_shape
    values=[]
    for name,p in actor.base_references:
        try:version=int(p._version)
        except RuntimeError:version=None
        values.append({'name':name,'shape':list(p.shape),'logical_shape':logical_shape(p),
            'dtype':str(p.dtype),'device':str(p.device),'requires_grad':p.requires_grad,
            'gradient_is_none':p.grad is None,'version':version,'sha256':tensor_digest(p)})
    return {'parameters':values,'auxiliary_sha256':auxiliary_digest(actor.base_references,actor.base_buffers),
            'live_recomputed':True,'original_backbone_only':True}


def audit_optimizer_bytes(raw,weights,updates,inspect=None):
    import torch
    values=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if set(values)!=set(weights):raise ValueError('Optimizer contains missing, extra or original backbone tensors')
    for name,shape in weights.items():
        value=values[name]
        if set(value)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Exact AdamW state required')
        step=value['step']
        if isinstance(step,torch.Tensor):
            if step.numel()!=1 or not torch.isfinite(step).all():raise ValueError('Invalid optimizer step tensor')
            step=step.item()
        if type(step) not in [int,float] or step!=updates:raise ValueError('AdamW step count differs')
        for key in ['exp_avg','exp_avg_sq']:
            tensor=value[key]
            if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or list(tensor.shape)!=shape or not torch.isfinite(tensor).all():
                raise ValueError('Raw AdamW moment geometry/precision/finite values differ')
        if (value['exp_avg_sq']<0).any():raise ValueError('AdamW second moments must be nonnegative')
        if inspect is not None:inspect(name,value)
    return {'raw_AdamW_tensors_verified':len(values)*2,'optimizer_step':updates}


def audit_runtime(value,live,original_shapes,*,expected_map=None,parameter_devices=None,require_native_bf16=True):
    from .composite_frozen_probe import verify_software,text_projection
    verify_software(value['torch'],value['packages'])
    expected_map={'':0} if expected_map is None else expected_map
    count=len(set(expected_map.values()))
    if (type(value.get('visible_GPUs')) is not int or value.get('visible_GPUs')!=count
        or type(value.get('BF16_supported')) is not bool or (require_native_bf16 and value.get('BF16_supported') is not True)
        or value.get('TF32_enabled') is not False or value.get('device_map')!=expected_map
        or value.get('kernel_profile')!='reference' or value.get('full_attention')!='sdpa'):
        raise ValueError('Pinned one-GPU reference runtime differs')
    quant=value['quantized_original_projections']
    live_by_name={v['name']:v for v in live['parameters']}
    packed={n for n,v in live_by_name.items() if v['dtype']=='torch.uint8'}
    if set(quant)!=packed or not {n for n in original_shapes if text_projection(n)}<=packed:
        raise ValueError('Complete original quantization coverage required')
    for name,proof in quant.items():
        device='cuda:0' if parameter_devices is None else parameter_devices[name]
        expected={'quant_type':'nf4','double_quantization':True,'storage_dtype':'torch.uint8',
            'compute_dtype':'torch.bfloat16','state_dtype':'torch.bfloat16','blocksize':64,'nested_blocksize':256,
            'logical_shape':original_shapes[name],'state_devices':{k:device for k in ['absmax','code','nested_absmax','nested_code']}}
        _same(proof,expected,'Pinned NF4 double-quant runtime differs')
    for name in original_shapes:
        if name.endswith('embed_tokens.weight') or name=='lm_head.weight':
            if live_by_name[name]['dtype']!='torch.bfloat16' or name in quant:raise ValueError('BF16 embedding/head changed')
    return {'software_and_original_quantization_verified':True}


def audit_bypass(files,token_plan,context):
    import numpy as np
    value=files.json('BYPASS_RAW_MANIFEST.json');records=value['records']
    selected=token_plan['qualification_state_ids'];lookup={r['state_id']:r for r in token_plan['records']}
    _same(value['logit_shape'],[1,1,context['vocabulary']],'Full pinned vocabulary logits required')
    if len(records)!=len(selected):raise ValueError('Both complete registered bypass prompts required')
    changed=False
    for index,(record,state) in enumerate(zip(records,selected)):
        _same(record['input_token_ids'],lookup[state]['input_token_ids'],'Bypass prompt differs')
        arrays=[]
        for name in ['wrapped_OFF','bare','wrapped_ON']:
            path='bypass/'+str(index).zfill(2)+'_'+name+'.npy'
            if record[name]['path']!=path:raise ValueError('Exact raw bypass namespace required')
            raw=files.get(path)
            if hashlib.sha256(raw).hexdigest()!=record[name]['sha256']:raise ValueError('Raw bypass logits changed')
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            if array.dtype!=np.float32 or array.shape!=tuple(value['logit_shape']) or not np.isfinite(array).all():
                raise ValueError('Complete finite raw bypass logits required')
            arrays.append(array)
        if not np.array_equal(arrays[0],arrays[1]):raise ValueError('Adapter OFF is not an exact bypass')
        changed=changed or not np.array_equal(arrays[0],arrays[2])
    if not changed:raise ValueError('No observed adapter ON effect')
    generation=files.json('BYPASS_CHECK.json')['records'][0]
    from .composite_v2_trace_audit import _native
    for name in ['wrapped_generation','bare_generation']:
        _native(generation[name],context['qualification_texts'][0],False,context['actor_tokenizer'],
                context['eos'],context['vocabulary'],None)
    _same(generation['wrapped_generation']['generated_token_ids'],generation['bare_generation']['generated_token_ids'],'OFF native generation differs')
    return {'raw_OFF_logits_exact':True,'raw_ON_difference_observed':True}


def frozen_integrity(before,after,expected_shapes,*,parameter_devices=None):
    for proof in [before,after]:
        if proof.get('live_recomputed') is not True or proof.get('original_backbone_only') is not True:
            raise ValueError('Cached snapshots cannot replace live tensor evidence')
        records=proof['parameters'];lookup={v['name']:v for v in records}
        if len(lookup)!=len(records) or set(lookup)!=set(expected_shapes):raise ValueError('Frozen parameter coverage differs')
        for name,value in lookup.items():
            _hash(value['sha256'])
            device='cuda:0' if parameter_devices is None else parameter_devices[name]
            if (value.get('requires_grad') is not False or value.get('gradient_is_none') is not True
                or value.get('device')!=device or digest(value.get('logical_shape'))!=digest(expected_shapes[name])):
                raise ValueError('Frozen original tensor geometry/gradient/device differs')
            if 'version' not in value or (value['version'] is not None and (type(value['version']) is not int or value['version']<0)):
                raise ValueError('Live parameter version availability must be explicit')
            shape=value.get('shape')
            if not isinstance(shape,list) or not shape or any(type(v) is not int or v<=0 for v in shape):
                raise ValueError('Recorded physical tensor geometry required')
            if value.get('dtype')=='torch.uint8':
                if math.prod(shape)!=(math.prod(expected_shapes[name])+1)//2:raise ValueError('Packed original tensor geometry differs')
            elif value.get('dtype') not in ['torch.float32','torch.bfloat16'] or shape!=expected_shapes[name]:
                raise ValueError('Original unquantized tensor precision/shape differs')
        _hash(proof['auxiliary_sha256'])
    _same(before,after,'Live original tensors or auxiliary state changed')
    return {'original_parameters_verified':len(expected_shapes),'live_bytes_recomputed_and_equal':True}


def checkpoint_evidence(raw,expected_sha256,cfg,projections,initial_hashes,*,rank=32):
    """Load adapter tensors only, with torch's restricted weights reader."""
    import torch
    from .composite_training import tensor_digest
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Checkpoint bytes changed')
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    _same(saved.get('config'),cfg,'Checkpoint architecture/configuration differs')
    if saved.get('native_interface_profile')!=PROFILE or saved.get('placement')!={'name':'all_blocks','scope':'all_blocks'}:
        raise ValueError('Checkpoint interface or adapter scope differs')
    expected={}
    for name,(inputs,outputs) in projections.items():expected.update({name+'.A':[rank,inputs],name+'.B':[outputs,rank]})
    weights=saved['weights']
    if set(weights)!=set(expected) or set(initial_hashes)!=set(expected):raise ValueError('Checkpoint tensor coverage differs')
    count=0;changed=False;hashes={}
    for name,value in weights.items():
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or list(value.shape)!=expected[name] or not torch.isfinite(value).all():
            raise ValueError('Complete finite FP32 adapter geometry required')
        _hash(initial_hashes[name]);hashes[name]=tensor_digest(value);count+=value.numel()
        changed=changed or hashes[name]!=initial_hashes[name]
    if not changed:raise ValueError('No independently observed adapter change')
    return {'adapter_parameters':count,'tensors':len(hashes),'hashes':hashes,'tensor_shapes':expected,'adapter_changed':True}


def _memory(value):
    names=['free_bytes','total_bytes','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes']
    if type(value.get('device')) is not int or value['device']!=0 or any(type(value.get(k)) is not int or value[k]<0 for k in names):raise ValueError('Raw device memory counters required')
    if not (0<=value['allocated_bytes']<=value['reserved_bytes']<=value['peak_reserved_bytes']<=value['total_bytes']
            and value['allocated_bytes']<=value['peak_allocated_bytes']<=value['peak_reserved_bytes']
            and 512*1024**2<=value['free_bytes']<=value['total_bytes']):raise ValueError('Runtime memory bounds differ')


def audit_model_files(files,context,token_plan,*,public_fixture=False,closure=None,_source_key=None):
    """Derive counts, checkpoint and integrity evidence from exact raw files.

    `context` is the independently source/input-verified model contract plus
    pinned header-derived projection and original-parameter shapes. Token plan
    is recomputed by the original tokenizer reader before this lower layer.
    This lower layer alone never issues a scientific qualification permit.
    """
    stage=context['stage'];source=context['source_commit'];phase=stage['phase']
    if phase not in ['qualification','full_fit'] or not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Exact model phase/source required')
    receipt=files.json('RUN_RECEIPT.json');blocks=[]
    _same(receipt.get('source_commit'),source,'Worker source differs')
    if receipt.get('phase')!=phase or receipt.get('native_interface_profile')!=PROFILE:raise ValueError('Worker phase/interface differs')
    if receipt.get('public_fixture_test_mode') is not False or public_fixture:blocks.append('public_fixture_not_scientific')
    if receipt.get('status')!='complete':blocks.append('worker_not_complete')
    if type(receipt.get('model_instances')) is not int or receipt['model_instances']!=1:blocks.append('one_model_instance_not_recorded')
    if receipt.get('counts_exact_at_controlled_exit') is not True:blocks.append('forward_counts_not_exact')
    required={'PRECALL_FREEZE.json','TRAINING_TOKEN_PLAN.json','BEHAVIOR_TRAINING.json','TRAINING_TERMINAL_JOURNAL.json',
              'behavior.pt','ADAPTER_INITIAL_HASHES.json'}
    missing=sorted(required-set(files.manifest))
    if missing:
        lower=None
        if 'TRAINING_TERMINAL_JOURNAL.json' in files.manifest:
            observed=files.json('TRAINING_TERMINAL_JOURNAL.json').get('optimizer_steps_completed')
            if type(observed) is int and 0<=observed<=(1 if phase=='qualification' else 64):lower=observed
        return ModelEvidence({'reader':READER,'source_commit':source,'phase':phase,'workspace':context['workspace'],
            'data_release_sha256':stage['data_release_sha256'],'raw_archive_sha256':files.sha256,
            'worker_status':receipt.get('status'),'missing_raw_files':missing,'missing_or_unverified':blocks,
            'recorded_update_count_verified':None,'optimizer_steps_completed_lower_bound':lower,
            'backbone_integrity':None,'model_tensor_integrity_audited':False,
            'complete_training_qualification_audited':False,'scientific_qualification_admitted':False,
            'owned_compute_released':None,'absence_of_unlogged_calls_proved':False,'public_fixture_test_mode':public_fixture,
            'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False},_KEY)
    _same(files.json('PRECALL_FREEZE.json'),{'stage':stage,'workspace':context['workspace'],
        'data_release_sha256':stage['data_release_sha256']},'Model/source/data/workspace precall freeze differs')
    _same(files.json('TRAINING_TOKEN_PLAN.json'),token_plan,'Complete tokenizer replay differs')
    training=files.json('BEHAVIOR_TRAINING.json');updates=1 if phase=='qualification' else 64
    counts=receipt.get('forward_counts',{})
    if (any(type(counts.get(k)) is not int for k in ['attempted','completed'])
        or not updates*2<=counts['completed']<=counts['attempted']<=30000):
        raise ValueError('Recorded controlled-exit forward counts differ from completed microsteps')
    expected_ids=token_plan['qualification_state_ids'] if phase=='qualification' else None
    if phase=='full_fit':
        import numpy as np
        rng=np.random.default_rng(202610212);rows=token_plan['records']
        expected_ids=[rows[int(rng.integers(len(rows)))]['state_id'] for _ in range(128)]
    if (type(training.get('optimizer_updates')) is not int or training['optimizer_updates']!=updates
        or training.get('heldout_content_read') is not False or receipt.get('heldout_content_read') is not False):
        raise ValueError('Exact phase update/held-out boundary differs')
    _same(training['sampled_state_ids'],expected_ids,'Seeded training sample identities differ')
    losses=training['losses']
    if len(losses)!=updates*2 or any(type(v) not in [int,float] or not math.isfinite(v) for v in losses):raise ValueError('Complete finite microstep losses required')
    names=set(context['projections']);gradient_names={n+'.'+p for n in names for p in ['A','B']}
    for index,state in enumerate(expected_ids):
        row=files.json('training/backward_'+str(index+1).zfill(4)+'.json')
        _same([row['update'],row['microstep'],row['state_id'],row['loss']],[index//2+1,index%2,state,losses[index]],'Backward journal differs')
        grads=row['gradients']
        if len(grads)!=len(gradient_names) or {g['name'] for g in grads}!=gradient_names:raise ValueError('Gradient projection coverage differs')
        if any(g.get('device')!='cuda:0' or g.get('dtype')!='torch.float32' or type(g.get('norm')) not in [int,float]
               or not math.isfinite(g['norm']) or g['norm']<0 for g in grads):raise ValueError('Finite FP32 CUDA gradients required')
        _memory(row['memory'])
    journal=files.json('TRAINING_TERMINAL_JOURNAL.json')
    for index in range(1,updates+1):
        row=files.json('training/update_'+str(index).zfill(3)+'.json');_memory(row['memory'])
        if any(type(row.get(k)) is not int or row[k]!=v for k,v in [('optimizer_steps_attempted',index),
            ('optimizer_steps_completed',index),('backward_microsteps_completed',index*2)]):
            raise ValueError('Update journal counts differ')
    if (any(type(journal.get(k)) is not int or journal[k]!=v for k,v in [('optimizer_steps_attempted',updates),
        ('optimizer_steps_completed',updates),('backward_microsteps_completed',updates*2)])
        or journal.get('in_backward') is not False):raise ValueError('Terminal training journal differs')
    for field in ['layer_forward_invocations','backward_recomputation_invocations']:
        if len(journal[field])!=64 or any(type(v) is not int or v<updates*2 for v in journal[field]):raise ValueError('All-block checkpoint/recomputation evidence missing')
    ck=checkpoint_evidence(files.get('behavior.pt'),training['checkpoint_sha256'],context['configuration']['qwen'],context['projections'],files.json('ADAPTER_INITIAL_HASHES.json'))
    if len(names)!=496 or ck['adapter_parameters']!=233455616:blocks.append('full_architecture_not_proved')
    available=set(files.manifest)
    integrity=None
    if {'BACKBONE_LIVE_BEFORE.json','BACKBONE_LIVE_AFTER.json'}<=available:
        integrity=frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),files.json('BACKBONE_LIVE_AFTER.json'),context['original_shapes'])
    else:blocks.append('live_backbone_bytes_and_flags_not_exported')
    # Current PR126 does not export complete AdamW moment/step tensor bytes,
    # live module state/configuration, or exact bypass logits for independent
    # recomputation. A worker assertion cannot stand in for these raw producers.
    if 'OPTIMIZER_FINAL.pt' in available:
        shapes={n+'.A':[32,v[0]] for n,v in context['projections'].items()}
        shapes.update({n+'.B':[v[1],32] for n,v in context['projections'].items()})
        audit_optimizer_bytes(files.get('OPTIMIZER_FINAL.pt'),shapes,updates)
    else:blocks.append('AdamW_raw_state_not_independently_audited')
    runtime=False
    if integrity is not None and 'RUNTIME_RAW.json' in available:
        audit_runtime(files.json('RUNTIME_RAW.json'),files.json('BACKBONE_LIVE_AFTER.json'),context['original_shapes']);runtime=True
    else:blocks.append('loaded_quantization_and_software_not_independently_audited')
    if 'BYPASS_RAW_MANIFEST.json' in available:audit_bypass(files,token_plan,context)
    else:blocks.append('raw_bypass_logits_not_independently_audited')
    released=(isinstance(closure,ProviderClosure) and closure._key is _KEY
        and closure.receipt['host']==context['workspace'].get('host') and closure.receipt['raw_archive_sha256']==files.sha256
        and type(receipt.get('provider_run_id')) is int and receipt['provider_run_id']==closure.receipt['run_id'])
    if not released:blocks.append('provider_terminal_release_not_independently_audited')
    if _source_key is not _KEY:blocks.append('source_model_input_and_tokenizer_bindings_not_verified_by_entry_point')
    complete=not blocks
    return ModelEvidence({'reader':READER,'source_commit':source,'phase':phase,
        'data_release_sha256':stage['data_release_sha256'],'workspace':context['workspace'],
        'raw_archive_sha256':files.sha256,'recorded_update_count_verified':updates,
        'checkpoint':ck,'checkpoint_sha256':training['checkpoint_sha256'],'backbone_integrity':integrity,'missing_or_unverified':blocks,
        'original_state_sha256':None if integrity is None else original_state_digest(files.json('BACKBONE_LIVE_AFTER.json')),
        'complete_model_stage_raw_audited':complete,'full_fit_raw_audited':complete and phase=='full_fit',
        'complete_training_qualification_audited':complete and phase=='qualification',
        'model_tensor_integrity_audited':integrity is not None and runtime and _source_key is _KEY and not public_fixture,
        'owned_compute_released':True if released else None,'provider_run_id':receipt.get('provider_run_id'),
        'checkpoint_initializes_full_fit':False,'absence_of_unlogged_calls_proved':False,
        'scientific_qualification_admitted':complete and phase=='qualification','public_fixture_test_mode':public_fixture,
        'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False},_KEY)


def audit_model_archive(raw,expected_sha256,root,source_commit,data_archive,*,closure=None,qualification_evidence=None):
    """Production entry: source, full model pin, header shapes and token replay.

    A currently incomplete PR126 export is diagnosable here but cannot acquire
    qualification by self-reporting success. No model weights are instantiated.
    """
    import base64,tempfile
    from pathlib import Path
    from .composite_v2_model_contract import validate_model_metadata,load_training_export
    from .composite_frozen_probe import validate_download_proof,verify_metadata_bytes
    from .composite_v2_training import token_plan as replay_token_plan
    from .io_utils import read,sha
    context=validate_model_metadata(root,source_commit)
    from .composite_generalized_data import require_runtime
    require_runtime()
    rows,data=load_training_export(data_archive,context,qualification_evidence=qualification_evidence)
    pin=read(Path(root)/'inputs/QWEN_PIN.json')
    metadata=read(Path(root)/'inputs/QWEN_TENSOR_METADATA.json')
    if (metadata.get('repository')!='Qwen/Qwen3.5-27B'
        or metadata.get('revision')!='fc05daec18b0a78c049392ed2e771dde82bdf654'
        or digest(metadata['tensors'])!='a2477ac2c15f9c198d5137204f00da050d8aa87716a3cc5f0156f00b84034f7f'):
        raise ValueError('Complete original header-derived tensor metadata required')
    originals={n:v['shape'] for n,v in metadata['tensors'].items() if not n.startswith('mtp.')}
    from .composite_frozen_probe import text_projection
    projections={n.removeprefix('model.language_model.').removesuffix('.weight'):[v[1],v[0]]
                 for n,v in originals.items() if text_projection(n)}
    if len(projections)!=496:raise ValueError('Complete pinned projection header coverage required')
    files=EvidenceFiles(raw,expected_sha256,context['stage'].get('maximum_raw_evidence_bytes',1024**3))
    try:
        full=files.json('MODEL_FILE_VERIFICATION.json');encoded=files.json('MODEL_METADATA_BYTES.json')
        validate_download_proof({'model_pin':pin},full);verify_metadata_bytes(pin,full,encoded)
        with tempfile.TemporaryDirectory(prefix='r4_model_audit_tokenizer_') as folder:
            target=Path(folder)
            for name,value in encoded.items():
                path=(target/name).resolve()
                if not path.is_relative_to(target.resolve()):raise ValueError('Metadata reconstruction path escaped')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value,validate=True))
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True,trust_remote_code=False)
            tokens=replay_token_plan(rows,tokenizer,context['configuration'])
            generation=read(target/'generation_config.json');eos=set()
            for value in [generation.get('eos_token_id'),tokenizer.eos_token_id]:
                if value is not None:eos.update(value if isinstance(value,list) else [value])
            by_id={r['state_id']:r for r in rows}
            extra={'actor_tokenizer':tokenizer,'eos':eos,'vocabulary':originals['lm_head.weight'][0],
                   'qualification_texts':[by_id[key]['text'] for key in tokens['qualification_state_ids']]}
        loading=files.json('MODEL_LOAD_ADMISSION.json')['loading']
        if (loading.get('missing_keys')!=[] or loading.get('mismatched_keys')!=[] or loading.get('error_msgs')!=[]
            or any(not n.startswith('mtp.') for n in loading['ignored_checkpoint_only_MTP_keys'])):
            raise ValueError('Model loader initialized missing or unexpected parameters')
        value=audit_model_files(files,{**context,**extra,'original_shapes':originals,'projections':projections},tokens,
                                closure=closure,_source_key=_KEY)
        # The raw reader does not convert missing runtime/optimizer/bypass or
        # provider-release evidence into a passed scientific qualification.
        receipt=value.receipt()
        receipt.update(source_release_sha256=sha(Path(root)/'RELEASE.json'),
                       source_input_manifest_sha256=sha(Path(root)/'inputs/MANIFEST.json'))
        return ModelEvidence(receipt,_KEY)
    finally:files.close()


def require_qualification_evidence(value,parent):
    if not isinstance(value,ModelEvidence) or value._key is not _KEY:raise PermissionError('Re-audited raw scratch qualification required')
    proof=value.receipt()
    if (proof.get('complete_training_qualification_audited') is not True or proof.get('scientific_qualification_admitted') is not True
        or proof.get('public_fixture_test_mode') is not False or proof.get('owned_compute_released') is not True
        or proof.get('phase')!='qualification' or proof.get('recorded_update_count_verified')!=1):
        raise PermissionError('Incomplete/public/unreleased evidence cannot admit a later full fit')
    for key in ['source_commit','data_release_sha256','workspace','raw_archive_sha256',
                'source_release_sha256','source_input_manifest_sha256']:
        _same(parent.get(key),proof[key],'Scratch qualification identity differs')
    return True


def detector_parameter_roster(files,pin,saved):
    """Shape-only meta architecture; no pretrained weights or forward call."""
    import base64,torch
    from transformers import AutoConfig,AutoModel
    from .composite_frozen_probe import verify_metadata_bytes
    from .composite_capacity import detector_targets
    proof=files.json('DETECTOR_FILE_VERIFICATION.json');encoded=files.json('DETECTOR_METADATA_BYTES.json')
    verify_metadata_bytes(pin,proof,encoded)
    config=json.loads(base64.b64decode(encoded['config.json'],validate=True))
    if config.pop('model_type')!='modernbert':raise ValueError('Pinned ModernBERT architecture required')
    with torch.device('meta'):
        encoder=AutoModel.from_config(AutoConfig.for_model('modernbert',**config),trust_remote_code=False,attn_implementation='sdpa')
    cfg=saved['config']
    if encoder.config.hidden_size!=cfg['hidden_size'] or encoder.config.num_hidden_layers!=cfg['layers']:
        raise ValueError('Detector header/checkpoint geometry differs')
    targets=detector_targets(encoder,cfg);roster={}
    for name in targets:
        module=encoder.get_submodule(name)
        for part,shape in [('A',[cfg['rank'],module.in_features]),('B',[module.out_features,cfg['rank']])]:
            tensor=saved['lora'].get(name+'.'+part)
            if tensor is None or list(tensor.shape)!=shape or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('Complete finite detector adapter geometry required')
    from .composite_detector import StructuredHeads
    with torch.device('meta'):heads=StructuredHeads(cfg['hidden_size'],saved.get('head_mode','cls'))
    if set(saved['heads'])!=set(heads.state_dict()) or any(list(saved['heads'][n].shape)!=list(v.shape)
        or saved['heads'][n].dtype!=torch.float32 or not torch.isfinite(saved['heads'][n]).all() for n,v in heads.state_dict().items()):
        raise ValueError('Complete selected detector head geometry required')
    for name,parameter in encoder.named_parameters():
        target=next((t for t in targets if name.startswith(t+'.')),None)
        key=name if target is None else target+'.base.'+name[len(target)+1:]
        roster['encoder.'+key]=list(parameter.shape)
    for prefix,key in [('encoder.','lora'),('heads.','heads')]:
        roster.update({prefix+n:list(v.shape) for n,v in saved[key].items()})
    expected_adapters={name+'.'+part for name in targets for part in ['A','B']}
    if set(saved['lora'])!=expected_adapters:raise ValueError('Complete detector projection coverage differs')
    return roster


def detector_integrity(files,pin,checkpoint,expected_sha256):
    """Verify frozen composite detector bytes and the selected adapter/heads."""
    import torch
    from .composite_training import tensor_digest
    if checkpoint is None or hashlib.sha256(checkpoint).hexdigest()!=expected_sha256:
        raise ValueError('Exact selected detector checkpoint bytes required')
    saved=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True)
    roster=detector_parameter_roster(files,pin,saved)
    before=files.json('DETECTOR_PARAMETERS_BEFORE.json');after=files.json('DETECTOR_PARAMETERS_AFTER.json')
    _same(before,after,'Validation detector changed')
    lookup={v['name']:v for v in before}
    if not before or len(lookup)!=len(before) or set(lookup)!=set(roster):raise ValueError('Complete unique frozen detector parameter records required')
    if any(v.get('requires_grad') is not False or v.get('gradient_is_none') is not True
           or v.get('device')!='cuda:0' or v.get('dtype')!='torch.float32'
           or v.get('shape')!=roster[v['name']] for v in before):
        raise ValueError('Detector parameters, adapters and heads must be frozen on the selected device')
    for value in before:_hash(value['sha256'])
    expected={**{'encoder.'+k:v for k,v in saved['lora'].items()},**{'heads.'+k:v for k,v in saved['heads'].items()}}
    for name,tensor in expected.items():
        value=lookup.get(name)
        if value is None or value.get('shape')!=list(tensor.shape) or value['sha256']!=tensor_digest(tensor):
            raise ValueError('Validation detector differs from the exact selected adapters/heads')
    proof=files.json('DETECTOR_FILE_VERIFICATION.json');by_name={v['name']:v for v in proof}
    if len(by_name)!=len(proof) or set(by_name)!={v['name'] for v in pin['files']}:raise ValueError('Original detector download coverage differs')
    for item in pin['files']:
        observed=by_name[item['name']];_hash(observed['sha256'])
        if observed['size']!=item['size'] or (item.get('sha256') and observed['sha256']!=item['sha256']):
            raise ValueError('Original detector file pin changed')
    runtime=files.json('DETECTOR_RUNTIME.json')
    expected_runtime={'config':saved['config'],'head_mode':saved.get('head_mode','cls'),
        'projection_names':saved['placements'],
        'projection_scales':{name:float(saved['config']['alpha'])/saved['config']['rank'] for name in saved['placements']},
        'all_modules_in_eval_mode':True,'full_attention':'sdpa'}
    _same(runtime,expected_runtime,'Actual frozen detector routing/runtime configuration differs')
    return {'frozen_parameters':len(before),'selected_checkpoint_tensors':len(expected),'checkpoint_sha256':expected_sha256}


def read_evaluation_integrity(files,root,source_commit,package,trace_manifest,model_stage_evidence,*,detector_checkpoint=None):
    """Independent frozen validation-state reader, not a training audit reused.

    Requires the actual audited full-fit identity and raw before/after adapter
    bytes and live original-tensor records from this validation stage. An old
    checkpoint or a caller boolean cannot assert validation runtime integrity.
    """
    if not isinstance(model_stage_evidence,ModelEvidence) or model_stage_evidence._key is not _KEY:
        raise PermissionError('Exact independently audited full-fit evidence required')
    prior=model_stage_evidence.receipt()
    if prior.get('full_fit_raw_audited') is not True or prior.get('public_fixture_test_mode') is not False:
        raise PermissionError('A scratch/public/incomplete fit cannot admit validation integrity')
    from pathlib import Path
    from .composite_v2_model_contract import validate_model_metadata
    from .composite_generalized_data import require_runtime
    from .composite_v2_admission import validate_package
    from .io_utils import read
    context=validate_model_metadata(root,source_commit);require_runtime();validate_package(package,context['configuration'])
    if (prior['source_commit']!=source_commit or prior['workspace']!=context['workspace']
        or prior['checkpoint_sha256']!=package['checkpoint_sha256']
        or prior['data_release_sha256']!=package['data_release_sha256']
        or digest(prior)!=package['training_raw_audit_sha256']):raise ValueError('Validation model package differs from the exact fit audit')
    binding=files.json('EVALUATION_MODEL_BINDING.json')
    _same(binding,{'source_commit':source_commit,'package_sha256':digest(package),
        'data_release_sha256':package['data_release_sha256'],'trace_manifest':trace_manifest,
        'optimizer_updates':0,'teacher_forcing':False,'decoder_repair':False,'policy_logit_mask':False,
        'public_fixture_test_mode':False},'Exact raw validation/source/trace binding differs')
    header=read(Path(root)/'inputs/QWEN_TENSOR_METADATA.json')
    if digest(header['tensors'])!='a2477ac2c15f9c198d5137204f00da050d8aa87716a3cc5f0156f00b84034f7f':
        raise ValueError('Pinned complete original tensor header differs')
    shapes={n:v['shape'] for n,v in header['tensors'].items() if not n.startswith('mtp.')}
    frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),files.json('BACKBONE_LIVE_AFTER.json'),shapes)
    if original_state_digest(files.json('BACKBONE_LIVE_AFTER.json'))!=prior.get('original_state_sha256'):
        raise ValueError('Validation original quantized bytes differ from the audited full fit')
    audit_runtime(files.json('RUNTIME_RAW.json'),files.json('BACKBONE_LIVE_AFTER.json'),shapes)
    design=read(Path(root)/'config/composite_generalization_amendment_v2.json')
    detector=detector_integrity(files,read(Path(root)/'inputs/MODERNBERT_PIN.json'),detector_checkpoint,
                                design['detector']['checkpoint_sha256'])
    import torch
    from .composite_training import tensor_digest
    for path in ['EVAL_ADAPTER_BEFORE.pt','EVAL_ADAPTER_AFTER.pt']:
        weights=torch.load(io.BytesIO(files.get(path)),map_location='cpu',weights_only=True)
        if set(weights)!=set(prior['checkpoint']['hashes']):raise ValueError('Evaluation adapter coverage differs')
        for name,value in weights.items():
            if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32
                or list(value.shape)!=prior['checkpoint']['tensor_shapes'][name]
                or not torch.isfinite(value).all() or tensor_digest(value)!=prior['checkpoint']['hashes'][name]):
                raise ValueError('Evaluation adapter bytes differ from the audited checkpoint')
    return EvaluationIntegrity({'reader':READER,'source_commit':source_commit,'package_sha256':digest(package),
        'trace_manifest_sha256':digest(trace_manifest),'trace_manifest':trace_manifest,'data_release_sha256':package['data_release_sha256'],
        'raw_archive_sha256':files.sha256,'detector_integrity':detector,'complete_model_integrity_audited':True,
        'absence_of_unlogged_calls_proved':False,'heldout_content_read':False},_KEY)


def bound_evaluation_integrity(value,source_commit,package_sha256,trace_manifest):
    if not isinstance(value,EvaluationIntegrity) or value._key is not _KEY:return False
    proof=value.receipt()
    return (proof['source_commit']==source_commit and proof['package_sha256']==package_sha256
            and proof['trace_manifest_sha256']==digest(trace_manifest) and proof['complete_model_integrity_audited'] is True)


def original_state_digest(proof):
    keys=['name','shape','logical_shape','dtype','sha256']
    return digest({'parameters':sorted([{k:r[k] for k in keys} for r in proof['parameters']],key=lambda r:r['name']),
                   'auxiliary_sha256':proof['auxiliary_sha256']})


def combine_evaluation_integrity(values,source_commit,package,trace_manifest):
    if not values or not trace_manifest:raise PermissionError('Nonempty independently audited partition integrity required')
    manifests={}
    for value in values:
        if not isinstance(value,EvaluationIntegrity) or value._key is not _KEY:raise PermissionError('Reader-produced partition integrity required')
        proof=value.receipt()
        if (proof['source_commit']!=source_commit or proof['package_sha256']!=digest(package)
            or proof['data_release_sha256']!=package['data_release_sha256'] or proof['complete_model_integrity_audited'] is not True):
            raise PermissionError('Integrity partitions belong to another exact freeze')
        if set(manifests)&set(proof['trace_manifest']):raise PermissionError('Overlapping integrity partitions')
        manifests.update(proof['trace_manifest'])
    if digest(manifests)!=digest(trace_manifest):raise PermissionError('Missing or mismatched partition integrity')
    return EvaluationIntegrity({'reader':READER,'source_commit':source_commit,'package_sha256':digest(package),
        'trace_manifest':manifests,'trace_manifest_sha256':digest(manifests),'data_release_sha256':package['data_release_sha256'],
        'complete_model_integrity_audited':True,'absence_of_unlogged_calls_proved':False,'heldout_content_read':False},_KEY)
